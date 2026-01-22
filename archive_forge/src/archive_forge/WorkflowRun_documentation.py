from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any, NamedTuple
import github.GitCommit
import github.PullRequest
import github.WorkflowJob
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional
from github.PaginatedList import PaginatedList

        :calls "`GET /repos/{owner}/{repo}/actions/runs/{run_id}/jobs <https://docs.github.com/en/rest/reference/actions#list-jobs-for-a-workflow-run>`_
        :param _filter: string `latest`, or `all`
        