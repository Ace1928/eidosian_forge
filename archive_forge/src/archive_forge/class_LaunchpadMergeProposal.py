import re
import shutil
import tempfile
from typing import Any, List, Optional
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ...forge import (AutoMergeUnsupported, Forge, LabelsUnsupported,
from ...git.urls import git_url_to_bzr_url
from ...lazy_import import lazy_import
from ...trace import mutter
from breezy.plugins.launchpad import (
from ...transport import get_transport
class LaunchpadMergeProposal(MergeProposal):
    supports_auto_merge = False

    def __init__(self, mp):
        self._mp = mp

    def get_web_url(self):
        return self._mp.web_link

    def get_source_branch_url(self, *, preferred_schemes=None):
        if self._mp.source_branch:
            return self._mp.source_branch.bzr_identity
        else:
            return git_url_to_bzr_url(self._mp.source_git_repository.git_identity, ref=self._mp.source_git_path.encode('utf-8'))

    def get_source_revision(self):
        if self._mp.source_branch:
            last_scanned_id = self._mp.source_branch.last_scanned_id
            if last_scanned_id:
                return last_scanned_id.encode('utf-8')
            else:
                return None
        else:
            from breezy.git.mapping import default_mapping
            git_repo = self._mp.source_git_repository
            git_ref = git_repo.getRefByPath(path=self._mp.source_git_path)
            sha = git_ref.commit_sha1
            if sha is None:
                return None
            return default_mapping.revision_id_foreign_to_bzr(sha.encode('ascii'))

    def get_target_branch_url(self, *, preferred_schemes=None):
        if self._mp.target_branch:
            return self._mp.target_branch.bzr_identity
        else:
            return git_url_to_bzr_url(self._mp.target_git_repository.git_identity, ref=self._mp.target_git_path.encode('utf-8'))

    def set_target_branch_name(self, name):
        raise NotImplementedError(self.set_target_branch_name)

    @property
    def url(self):
        return lp_uris.canonical_url(self._mp)

    def is_merged(self):
        return self._mp.queue_status == 'Merged'

    def is_closed(self):
        return self._mp.queue_status in ('Rejected', 'Superseded')

    def reopen(self):
        self._mp.setStatus(status='Needs review')

    def get_description(self):
        return self._mp.description

    def set_description(self, description):
        self._mp.description = description
        self._mp.lp_save()

    def get_commit_message(self):
        return self._mp.commit_message

    def get_title(self):
        raise TitleUnsupported(self)

    def set_title(self):
        raise TitleUnsupported(self)

    def set_commit_message(self, commit_message):
        self._mp.commit_message = commit_message
        self._mp.lp_save()

    def close(self):
        self._mp.setStatus(status='Rejected')

    def can_be_merged(self):
        if not self._mp.preview_diff:
            return True
        return not bool(self._mp.preview_diff.conflicts)

    def get_merged_by(self):
        merge_reporter = self._mp.merge_reporter
        if merge_reporter is None:
            return None
        return merge_reporter.name

    def get_merged_at(self):
        return self._mp.date_merged

    def merge(self, commit_message=None, auto=False):
        if auto:
            raise AutoMergeUnsupported(self)
        target_branch = _mod_branch.Branch.open(self.get_target_branch_url())
        source_branch = _mod_branch.Branch.open(self.get_source_branch_url())
        tmpdir = tempfile.mkdtemp()
        try:
            tree = target_branch.create_checkout(to_location=tmpdir, lightweight=True)
            tree.merge_from_branch(source_branch)
            tree.commit(commit_message or self._mp.commit_message)
        finally:
            shutil.rmtree(tmpdir)

    def post_comment(self, body):
        self._mp.createComment(content=body)