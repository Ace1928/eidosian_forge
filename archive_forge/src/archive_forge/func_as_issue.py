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
def as_issue(self) -> Issue:
    """
        :calls: `GET /repos/{owner}/{repo}/issues/{number} <https://docs.github.com/en/rest/reference/issues>`_
        """
    headers, data = self._requester.requestJsonAndCheck('GET', self.issue_url)
    return github.Issue.Issue(self._requester, headers, data, completed=True)