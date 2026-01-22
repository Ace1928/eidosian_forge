from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.NamedUser
import github.ProjectColumn
from github import Consts
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList

        calls: `POST /projects/{project_id}/columns <https://docs.github.com/en/rest/reference/projects#create-a-project-column>`_
        