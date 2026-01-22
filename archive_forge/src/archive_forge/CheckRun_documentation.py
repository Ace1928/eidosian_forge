from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.CheckRunAnnotation
import github.CheckRunOutput
import github.GithubApp
import github.GithubObject
import github.PullRequest
from github.GithubObject import (
from github.PaginatedList import PaginatedList

        :calls: `PATCH /repos/{owner}/{repo}/check-runs/{check_run_id} <https://docs.github.com/en/rest/reference/checks#update-a-check-run>`_
        