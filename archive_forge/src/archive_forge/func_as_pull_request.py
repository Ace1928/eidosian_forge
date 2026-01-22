from __future__ import annotations
import urllib.parse
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.GithubObject
import github.IssueComment
import github.IssueEvent
import github.IssuePullRequest
import github.Label
import github.Milestone
import github.NamedUser
import github.PullRequest
import github.Reaction
import github.Repository
import github.TimelineEvent
from github import Consts
from github.GithubObject import (
from github.PaginatedList import PaginatedList
def as_pull_request(self) -> PullRequest:
    """
        :calls: `GET /repos/{owner}/{repo}/pulls/{number} <https://docs.github.com/en/rest/reference/pulls>`_
        """
    headers, data = self._requester.requestJsonAndCheck('GET', '/pulls/'.join(self.url.rsplit('/issues/', 1)))
    return github.PullRequest.PullRequest(self._requester, headers, data, completed=True)