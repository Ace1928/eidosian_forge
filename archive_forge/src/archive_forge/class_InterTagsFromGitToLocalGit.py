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
class InterTagsFromGitToLocalGit(InterTags):

    @classmethod
    def is_compatible(klass, source, target):
        if not isinstance(source, GitTags):
            return False
        if not isinstance(target, GitTags):
            return False
        if getattr(target.branch.repository, '_git', None) is None:
            return False
        return True

    def merge(self, overwrite=False, ignore_master=False, selector=None):
        if self.source.branch.repository.has_same_location(self.target.branch.repository):
            return ({}, [])
        conflicts = []
        updates = {}
        source_tag_refs = self.source.branch.get_tag_refs()
        target_repo = self.target.branch.repository
        for ref_name, tag_name, peeled, unpeeled in source_tag_refs:
            if selector and (not selector(tag_name)):
                continue
            if target_repo._git.refs.get(ref_name) == unpeeled:
                pass
            elif overwrite or ref_name not in target_repo._git.refs:
                try:
                    updates[tag_name] = target_repo.lookup_foreign_revision_id(peeled)
                except KeyError:
                    trace.warning('%s does not point to a valid object', tag_name)
                    continue
                except NotCommitError:
                    trace.warning('%s points to a non-commit object', tag_name)
                    continue
                target_repo._git.refs[ref_name] = unpeeled or peeled
                self.target.branch._tag_refs = None
            else:
                try:
                    source_revid = self.source.branch.repository.lookup_foreign_revision_id(peeled)
                    target_revid = target_repo.lookup_foreign_revision_id(target_repo._git.refs[ref_name])
                except KeyError:
                    trace.warning('%s does not point to a valid object', ref_name)
                    continue
                except NotCommitError:
                    trace.warning('%s points to a non-commit object', tag_name)
                    continue
                conflicts.append((tag_name, source_revid, target_revid))
        return (updates, set(conflicts))