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
class ProjectUserManager(ListMixin, RESTManager):
    _path = '/projects/{project_id}/users'
    _obj_cls = ProjectUser
    _from_parent_attrs = {'project_id': 'id'}
    _list_filters = ('search', 'skip_users')
    _types = {'skip_users': types.ArrayAttribute}