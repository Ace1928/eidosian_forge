from __future__ import annotations
from datetime import datetime
from typing import TYPE_CHECKING, Any
import github.EnvironmentDeploymentBranchPolicy
import github.EnvironmentProtectionRule
from github.GithubObject import Attribute, CompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
from github.PublicKey import PublicKey
from github.Secret import Secret
from github.Variable import Variable

        :calls: `DELETE /repositories/{repository_id}/environments/{environment_name}/variables/{variable_name} <https://docs.github.com/en/rest/reference#delete-a-repository-variable>`_
        :param variable_name: string
        :rtype: bool
        