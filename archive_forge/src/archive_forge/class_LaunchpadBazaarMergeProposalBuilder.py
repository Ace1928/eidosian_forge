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
class LaunchpadBazaarMergeProposalBuilder(MergeProposalBuilder):

    def __init__(self, lp_host, source_branch, target_branch, staging=None, approve=None, fixes=None):
        """Constructor.

        :param source_branch: The branch to propose for merging.
        :param target_branch: The branch to merge into.
        :param staging: If True, propose the merge against staging instead of
            production.
        :param approve: If True, mark the new proposal as approved immediately.
            This is useful when a project permits some things to be approved
            by the submitter (e.g. merges between release and deployment
            branches).
        """
        self.lp_host = lp_host
        self.launchpad = lp_host.launchpad
        self.source_branch = source_branch
        self.source_branch_lp = self.launchpad.branches.getByUrl(url=source_branch.user_url)
        if target_branch is None:
            self.target_branch_lp = self.source_branch_lp.get_target()
            self.target_branch = _mod_branch.Branch.open(self.target_branch_lp.bzr_identity)
        else:
            self.target_branch = target_branch
            self.target_branch_lp = self.launchpad.branches.getByUrl(url=target_branch.user_url)
        self.approve = approve
        self.fixes = fixes

    def get_infotext(self):
        """Determine the initial comment for the merge proposal."""
        info = ['Source: %s\n' % self.source_branch_lp.bzr_identity]
        info.append('Target: %s\n' % self.target_branch_lp.bzr_identity)
        return ''.join(info)

    def get_initial_body(self):
        """Get a body for the proposal for the user to modify.

        :return: a str or None.
        """
        if not self.hooks['merge_proposal_body']:
            return None

        def list_modified_files():
            lca_tree = self.source_branch_lp.find_lca_tree(self.target_branch_lp)
            source_tree = self.source_branch.basis_tree()
            files = modified_files(lca_tree, source_tree)
            return list(files)
        with self.target_branch.lock_read(), self.source_branch.lock_read():
            body = None
            for hook in self.hooks['merge_proposal_body']:
                body = hook({'target_branch': self.target_branch_lp.bzr_identity, 'modified_files_callback': list_modified_files, 'old_body': body})
            return body

    def check_proposal(self):
        """Check that the submission is sensible."""
        if self.source_branch_lp.self_link == self.target_branch_lp.self_link:
            raise errors.CommandError('Source and target branches must be different.')
        for mp in self.source_branch_lp.landing_targets:
            if mp.queue_status in ('Merged', 'Rejected'):
                continue
            if mp.target_branch.self_link == self.target_branch_lp.self_link:
                raise MergeProposalExists(lp_uris.canonical_url(mp))

    def approve_proposal(self, mp):
        with self.source_branch.lock_read():
            _call_webservice(mp.createComment, vote='Approve', subject='', content='Rubberstamp! Proposer approves of own proposal.')
            _call_webservice(mp.setStatus, status='Approved', revid=self.source_branch.last_revision())

    def create_proposal(self, description, title=None, reviewers=None, labels=None, prerequisite_branch=None, commit_message=None, work_in_progress=False, allow_collaboration=False, delete_source_after_merge: Optional[bool]=None):
        """Perform the submission."""
        if labels:
            raise LabelsUnsupported(self)
        if title:
            raise TitleUnsupported(self)
        if prerequisite_branch is not None:
            prereq = self.launchpad.branches.getByUrl(url=prerequisite_branch.user_url)
        else:
            prereq = None
        if reviewers is None:
            reviewer_objs: List[Any] = []
        else:
            reviewer_objs = []
            for reviewer in reviewers:
                reviewer_objs.append(self.lp_host._getPerson(reviewer))
        if delete_source_after_merge is True:
            mutter('Ignoring request to delete source after merge, which launchpad does not support')
        try:
            mp = _call_webservice(self.source_branch_lp.createMergeProposal, target_branch=self.target_branch_lp, prerequisite_branch=prereq, initial_comment=description.strip(), commit_message=commit_message, needs_review=not work_in_progress, reviewers=[reviewer.self_link for reviewer in reviewer_objs], review_types=['' for reviewer in reviewer_objs])
        except WebserviceFailure as e:
            if b'There is already a branch merge proposal registered for branch ' in e.message:
                raise MergeProposalExists(self.source_branch.user_url)
            raise
        if self.approve:
            self.approve_proposal(mp)
        if self.fixes:
            if self.fixes.startswith('lp:'):
                self.fixes = self.fixes[3:]
            _call_webservice(mp.linkBug, bug=self.launchpad.bugs[int(self.fixes)])
        return LaunchpadMergeProposal(mp)