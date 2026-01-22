from typing import Any, cast, Dict, List, Optional, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
from .custom_attributes import UserCustomAttributeManager  # noqa: F401
from .events import UserEventManager  # noqa: F401
from .personal_access_tokens import UserPersonalAccessTokenManager  # noqa: F401
class UserProjectManager(ListMixin, CreateMixin, RESTManager):
    _path = '/projects/user/{user_id}'
    _obj_cls = UserProject
    _from_parent_attrs = {'user_id': 'id'}
    _create_attrs = RequiredOptional(required=('name',), optional=('default_branch', 'issues_enabled', 'wall_enabled', 'merge_requests_enabled', 'wiki_enabled', 'snippets_enabled', 'squash_option', 'public', 'visibility', 'description', 'builds_enabled', 'public_builds', 'import_url', 'only_allow_merge_if_build_succeeds'))
    _list_filters = ('archived', 'visibility', 'order_by', 'sort', 'search', 'simple', 'owned', 'membership', 'starred', 'statistics', 'with_issues_enabled', 'with_merge_requests_enabled', 'with_custom_attributes', 'with_programming_language', 'wiki_checksum_failed', 'repository_checksum_failed', 'min_access_level', 'id_after', 'id_before')

    def list(self, **kwargs: Any) -> Union[RESTObjectList, List[RESTObject]]:
        """Retrieve a list of objects.

        Args:
            all: If True, return all the items, without pagination
            per_page: Number of items to retrieve per request
            page: ID of the page to return (starts with page 1)
            iterator: If set to True and no pagination option is
                defined, return a generator instead of a list
            **kwargs: Extra options to send to the server (e.g. sudo)

        Returns:
            The list of objects, or a generator if `iterator` is True

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabListError: If the server cannot perform the request
        """
        if self._parent:
            path = f'/users/{self._parent.id}/projects'
        else:
            path = f'/users/{self._from_parent_attrs['user_id']}/projects'
        return ListMixin.list(self, path=path, **kwargs)