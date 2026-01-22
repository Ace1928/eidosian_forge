from datetime import datetime
from typing import Any, Dict
from github.GithubObject import Attribute, NotSet
from github.PaginatedList import PaginatedList
from github.Repository import Repository
from github.Secret import Secret
def add_repo(self, repo: Repository) -> bool:
    """
        :calls: 'PUT {org_url}/actions/secrets/{secret_name} <https://docs.github.com/en/rest/actions/secrets#add-selected-repository-to-an-organization-secret>`_
        :param repo: github.Repository.Repository
        :rtype: bool
        """
    if self.visibility != 'selected':
        return False
    self._requester.requestJsonAndCheck('PUT', f'{self._selected_repositories_url.value}/{repo.id}')
    return True