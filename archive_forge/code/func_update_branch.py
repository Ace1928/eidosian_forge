from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import NotRequired, TypedDict
import github.Commit
import github.File
import github.IssueComment
import github.IssueEvent
import github.Label
import github.Milestone
import github.NamedUser
import github.PaginatedList
import github.PullRequestComment
import github.PullRequestMergeStatus
import github.PullRequestPart
import github.PullRequestReview
import github.Team
from github import Consts
from github.GithubObject import (
from github.Issue import Issue
from github.PaginatedList import PaginatedList
def update_branch(self, expected_head_sha: Opt[str]=NotSet) -> bool:
    """
        :calls `PUT /repos/{owner}/{repo}/pulls/{pull_number}/update-branch <https://docs.github.com/en/rest/reference/pulls>`_
        """
    assert is_optional(expected_head_sha, str), expected_head_sha
    post_parameters = NotSet.remove_unset_items({'expected_head_sha': expected_head_sha})
    status, headers, data = self._requester.requestJson('PUT', f'{self.url}/update-branch', input=post_parameters, headers={'Accept': Consts.updateBranchPreview})
    return status == 202