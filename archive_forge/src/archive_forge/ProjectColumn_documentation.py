from __future__ import annotations
from datetime import datetime
from typing import Any
import github.GithubObject
import github.Project
import github.ProjectCard
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
from . import Consts

        :calls: `PATCH /projects/columns/{column_id} <https://docs.github.com/en/rest/reference/projects#update-an-existing-project-column>`_
        