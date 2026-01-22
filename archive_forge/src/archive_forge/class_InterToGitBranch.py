import contextlib
from collections import defaultdict
from functools import partial
from io import BytesIO
from typing import Dict, Optional, Set, Tuple
from dulwich.config import ConfigFile as GitConfigFile
from dulwich.config import parse_submodules
from dulwich.objects import ZERO_SHA, NotCommitError
from dulwich.repo import check_ref_format
from .. import branch, config, controldir, errors, lock
from .. import repository as _mod_repository
from .. import revision, trace, transport, urlutils
from ..foreign import ForeignBranch
from ..revision import NULL_REVISION
from ..tag import InterTags, TagConflict, Tags, TagSelector, TagUpdates
from ..trace import is_quiet, mutter, warning
from .errors import NoPushSupport
from .mapping import decode_git_path, encode_git_path
from .push import remote_divergence
from .refs import (branch_name_to_ref, is_tag, ref_to_branch_name,
from .unpeel_map import UnpeelMap
from .urls import bzr_url_to_git_url, git_url_to_bzr_url
class InterToGitBranch(branch.GenericInterBranch):
    """InterBranch implementation that pulls into a Git branch."""

    def __init__(self, source, target):
        super().__init__(source, target)
        self.interrepo = _mod_repository.InterRepository.get(source.repository, target.repository)

    @staticmethod
    def _get_branch_formats_to_test():
        try:
            default_format = branch.format_registry.get_default()
        except AttributeError:
            default_format = branch.BranchFormat._default_format
        from .remote import RemoteGitBranchFormat
        return [(default_format, LocalGitBranchFormat()), (default_format, RemoteGitBranchFormat())]

    @classmethod
    def is_compatible(self, source, target):
        return not isinstance(source, GitBranch) and isinstance(target, GitBranch)

    def _get_new_refs(self, stop_revision=None, fetch_tags=None, stop_revno=None):
        if not self.source.is_locked():
            raise errors.ObjectNotLocked(self.source)
        if stop_revision is None:
            stop_revno, stop_revision = self.source.last_revision_info()
        elif stop_revno is None:
            try:
                stop_revno = self.source.revision_id_to_revno(stop_revision)
            except errors.NoSuchRevision:
                stop_revno = None
        if not isinstance(stop_revision, bytes):
            raise TypeError(stop_revision)
        main_ref = self.target.ref
        refs = {main_ref: (None, stop_revision)}
        if fetch_tags is None:
            c = self.source.get_config_stack()
            fetch_tags = c.get('branch.fetch_tags')
        for name, revid in self.source.tags.get_tag_dict().items():
            if self.source.repository.has_revision(revid):
                ref = tag_name_to_ref(name)
                if not check_ref_format(ref):
                    warning('skipping tag with invalid characters %s (%s)', name, ref)
                    continue
                if fetch_tags:
                    refs[ref] = (None, revid)
        return (refs, main_ref, (stop_revno, stop_revision))

    def fetch(self, stop_revision=None, fetch_tags=None, lossy=False, limit=None):
        if stop_revision is None:
            stop_revision = self.source.last_revision()
        ret = []
        if fetch_tags:
            for k, v in self.source.tags.get_tag_dict().items():
                ret.append((None, v))
        ret.append((None, stop_revision))
        if getattr(self.interrepo, 'fetch_revs', None):
            try:
                revidmap = self.interrepo.fetch_revs(ret, lossy=lossy, limit=limit)
            except NoPushSupport:
                raise errors.NoRoundtrippingSupport(self.source, self.target)
            return _mod_repository.FetchResult(revidmap={old_revid: new_revid for old_revid, (new_sha, new_revid) in revidmap.items()})
        else:

            def determine_wants(refs):
                wants = []
                for git_sha, revid in ret:
                    if git_sha is None:
                        git_sha, mapping = self.target.lookup_bzr_revision_id(revid)
                    wants.append(git_sha)
                return wants
            self.interrepo.fetch_objects(determine_wants, lossy=lossy, limit=limit)
            return _mod_repository.FetchResult()

    def pull(self, overwrite=False, stop_revision=None, local=False, possible_transports=None, run_hooks=True, _stop_revno=None, tag_selector=None):
        result = GitBranchPullResult()
        result.source_branch = self.source
        result.target_branch = self.target
        with self.source.lock_read(), self.target.lock_write():
            new_refs, main_ref, stop_revinfo = self._get_new_refs(stop_revision, stop_revno=_stop_revno)
            update_refs = partial(_update_pure_git_refs, result, new_refs, overwrite, tag_selector)
            try:
                result.revidmap, old_refs, new_refs = self.interrepo.fetch_refs(update_refs, lossy=False)
            except NoPushSupport:
                raise errors.NoRoundtrippingSupport(self.source, self.target)
            old_sha1, result.old_revid = old_refs.get(main_ref, (ZERO_SHA, NULL_REVISION))
            if result.old_revid is None:
                result.old_revid = self.target.lookup_foreign_revision_id(old_sha1)
            result.new_revid = new_refs[main_ref][1]
            result.local_branch = None
            result.master_branch = self.target
            if run_hooks:
                for hook in branch.Branch.hooks['post_pull']:
                    hook(result)
        return result

    def push(self, overwrite=False, stop_revision=None, lossy=False, _override_hook_source_branch=None, _stop_revno=None, tag_selector=None):
        result = GitBranchPushResult()
        result.source_branch = self.source
        result.target_branch = self.target
        result.local_branch = None
        result.master_branch = result.target_branch
        with self.source.lock_read(), self.target.lock_write():
            new_refs, main_ref, stop_revinfo = self._get_new_refs(stop_revision, stop_revno=_stop_revno)
            update_refs = partial(_update_pure_git_refs, result, new_refs, overwrite, tag_selector)
            try:
                result.revidmap, old_refs, new_refs = self.interrepo.fetch_refs(update_refs, lossy=lossy, overwrite=overwrite)
            except NoPushSupport:
                raise errors.NoRoundtrippingSupport(self.source, self.target)
            old_sha1, result.old_revid = old_refs.get(main_ref, (ZERO_SHA, NULL_REVISION))
            if lossy or result.old_revid is None:
                result.old_revid = self.target.lookup_foreign_revision_id(old_sha1)
            result.new_revid = new_refs[main_ref][1]
            result.new_original_revno, result.new_original_revid = stop_revinfo
            for hook in branch.Branch.hooks['post_push']:
                hook(result)
        return result