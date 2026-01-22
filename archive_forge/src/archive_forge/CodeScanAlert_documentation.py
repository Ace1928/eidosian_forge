from __future__ import annotations
from datetime import datetime
from typing import Any
import github.CodeScanAlertInstance
import github.CodeScanRule
import github.CodeScanTool
import github.GithubObject
import github.NamedUser
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList

        :calls: `GET` on the URL for instances as provided by Github
        