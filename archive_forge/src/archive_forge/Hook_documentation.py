from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.HookResponse
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt, is_optional, is_optional_list

        :calls: `POST /repos/{owner}/{repo}/hooks/{id}/pings <https://docs.github.com/en/rest/reference/repos#webhooks>`_
        