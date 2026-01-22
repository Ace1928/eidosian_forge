from __future__ import annotations
from datetime import datetime
from typing import Any
import github.Consts
import github.DeploymentStatus
import github.NamedUser
from github.GithubObject import Attribute, CompletableGithubObject, NotSet, Opt
from github.PaginatedList import PaginatedList
@staticmethod
def _get_accept_header() -> str:
    return ', '.join([github.Consts.deploymentEnhancementsPreview, github.Consts.deploymentStatusEnhancementsPreview])