from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
class PersonalAccessTokenManager(DeleteMixin, RetrieveMixin, RotateMixin, RESTManager):
    _path = '/personal_access_tokens'
    _obj_cls = PersonalAccessToken
    _list_filters = ('user_id',)

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> PersonalAccessToken:
        return cast(PersonalAccessToken, super().get(id=id, lazy=lazy, **kwargs))