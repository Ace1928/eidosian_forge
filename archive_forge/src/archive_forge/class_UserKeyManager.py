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
class UserKeyManager(RetrieveMixin, CreateMixin, DeleteMixin, RESTManager):
    _path = '/users/{user_id}/keys'
    _obj_cls = UserKey
    _from_parent_attrs = {'user_id': 'id'}
    _create_attrs = RequiredOptional(required=('title', 'key'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> UserKey:
        return cast(UserKey, super().get(id=id, lazy=lazy, **kwargs))