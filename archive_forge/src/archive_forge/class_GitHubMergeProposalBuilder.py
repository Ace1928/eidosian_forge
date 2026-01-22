import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from ... import bedding
from ... import branch as _mod_branch
from ... import controldir, errors, hooks, urlutils
from ... import version_string as breezy_version
from ...config import AuthenticationConfig, GlobalStack
from ...errors import (InvalidHttpResponse, PermissionDenied,
from ...forge import (Forge, ForgeLoginRequired, MergeProposal,
from ...git.urls import git_url_to_bzr_url
from ...i18n import gettext
from ...trace import note
from ...transport import get_transport
from ...transport.http import default_user_agent
class GitHubMergeProposalBuilder(MergeProposalBuilder):

    def __init__(self, gh, source_branch, target_branch):
        self.gh = gh
        self.source_branch = source_branch
        self.target_branch = target_branch
        self.target_owner, self.target_repo_name, self.target_branch_name = parse_github_branch_url(self.target_branch)
        self.source_owner, self.source_repo_name, self.source_branch_name = parse_github_branch_url(self.source_branch)

    def get_infotext(self):
        """Determine the initial comment for the merge proposal."""
        info = []
        info.append('Merge {} into {}:{}\n'.format(self.source_branch_name, self.target_owner, self.target_branch_name))
        info.append('Source: %s\n' % self.source_branch.user_url)
        info.append('Target: %s\n' % self.target_branch.user_url)
        return ''.join(info)

    def get_initial_body(self):
        """Get a body for the proposal for the user to modify.

        :return: a str or None.
        """
        return None

    def create_proposal(self, description, title=None, reviewers=None, labels=None, prerequisite_branch=None, commit_message=None, work_in_progress=False, allow_collaboration=False, delete_source_after_merge: Optional[bool]=None):
        """Perform the submission."""
        if prerequisite_branch is not None:
            raise PrerequisiteBranchUnsupported(self)
        if self.target_repo_name.endswith('.git'):
            self.target_repo_name = self.target_repo_name[:-4]
        if title is None:
            title = determine_title(description)
        target_repo = self.gh._get_repo(self.target_owner, self.target_repo_name)
        assignees: Optional[List[Dict[str, Any]]] = []
        if reviewers:
            assignees = []
            for reviewer in reviewers:
                if '@' in reviewer:
                    user = self.gh._get_user_by_email(reviewer)
                else:
                    user = self.gh._get_user(reviewer)
                assignees.append(user['login'])
        else:
            assignees = None
        kwargs: Dict[str, Any] = {}
        if delete_source_after_merge is not None:
            kwargs['delete_branch_on_merge'] = delete_source_after_merge
        try:
            pull_request = self.gh._create_pull(strip_optional(target_repo['pulls_url']), title=title, body=description, head='{}:{}'.format(self.source_owner, self.source_branch_name), base=self.target_branch_name, labels=labels, assignee=assignees, draft=work_in_progress, maintainer_can_modify=allow_collaboration, **kwargs)
        except ValidationFailed:
            raise MergeProposalExists(self.source_branch.user_url)
        return GitHubMergeProposal(self.gh, pull_request)