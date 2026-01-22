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
class StarredProjectManager(ListMixin, RESTManager):
    _path = '/users/{user_id}/starred_projects'
    _obj_cls = StarredProject
    _from_parent_attrs = {'user_id': 'id'}
    _list_filters = ('archived', 'membership', 'min_access_level', 'order_by', 'owned', 'search', 'simple', 'sort', 'starred', 'statistics', 'visibility', 'with_custom_attributes', 'with_issues_enabled', 'with_merge_requests_enabled')