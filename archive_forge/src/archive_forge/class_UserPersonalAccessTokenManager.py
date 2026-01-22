from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
class UserPersonalAccessTokenManager(CreateMixin, RESTManager):
    _path = '/users/{user_id}/personal_access_tokens'
    _obj_cls = UserPersonalAccessToken
    _from_parent_attrs = {'user_id': 'id'}
    _create_attrs = RequiredOptional(required=('name', 'scopes'), optional=('expires_at',))
    _types = {'scopes': ArrayAttribute}