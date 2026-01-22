from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, NotSet
from github.PaginatedList import PaginatedList
from github.Repository import Repository
from github.Variable import Variable

        :calls: 'DELETE {org_url}/actions/variables/{variable_name} <https://docs.github.com/en/rest/actions/variables#add-selected-repository-to-an-organization-secret>`_
        :param repo: github.Repository.Repository
        :rtype: bool
        