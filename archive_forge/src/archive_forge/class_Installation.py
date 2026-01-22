from __future__ import annotations
from typing import TYPE_CHECKING, Any
import github.Authorization
import github.Event
import github.Gist
import github.GithubObject
import github.Issue
import github.Notification
import github.Organization
import github.PaginatedList
import github.Plan
import github.Repository
import github.UserKey
from github import Consts
from github.Auth import AppAuth
from github.GithubObject import Attribute, NonCompletableGithubObject, NotSet
from github.PaginatedList import PaginatedList
from github.Requester import Requester
class Installation(NonCompletableGithubObject):
    """
    This class represents Installations. The reference can be found here https://docs.github.com/en/rest/reference/apps#installations
    """

    def __init__(self, requester: Requester, headers: dict[str, str | int], attributes: Any, completed: bool) -> None:
        super().__init__(requester, headers, attributes, completed)
        auth = self._requester.auth if self._requester is not None else None
        if isinstance(auth, AppAuth):
            auth = auth.get_installation_auth(self.id, requester=self._requester)
            self._requester = self._requester.withAuth(auth)

    def _initAttributes(self) -> None:
        self._id: Attribute[int] = NotSet
        self._app_id: Attribute[int] = NotSet
        self._target_id: Attribute[int] = NotSet
        self._target_type: Attribute[str] = NotSet

    def __repr__(self) -> str:
        return self.get__repr__({'id': self._id.value})

    def get_github_for_installation(self) -> Github:
        return github.Github(**self._requester.kwargs)

    @property
    def id(self) -> int:
        return self._id.value

    @property
    def app_id(self) -> int:
        return self._app_id.value

    @property
    def target_id(self) -> int:
        return self._target_id.value

    @property
    def target_type(self) -> str:
        return self._target_type.value

    def get_repos(self) -> PaginatedList[github.Repository.Repository]:
        """
        :calls: `GET /installation/repositories <https://docs.github.com/en/rest/reference/integrations/installations#list-repositories>`_
        """
        url_parameters: dict[str, Any] = {}
        return PaginatedList(contentClass=github.Repository.Repository, requester=self._requester, firstUrl='/installation/repositories', firstParams=url_parameters, headers=INTEGRATION_PREVIEW_HEADERS, list_item='repositories')

    def _useAttributes(self, attributes: dict[str, Any]) -> None:
        if 'id' in attributes:
            self._id = self._makeIntAttribute(attributes['id'])
        if 'app_id' in attributes:
            self._app_id = self._makeIntAttribute(attributes['app_id'])
        if 'target_id' in attributes:
            self._target_id = self._makeIntAttribute(attributes['target_id'])
        if 'target_type' in attributes:
            self._target_type = self._makeStringAttribute(attributes['target_type'])