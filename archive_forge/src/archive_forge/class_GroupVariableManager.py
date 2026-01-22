from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class GroupVariableManager(CRUDMixin, RESTManager):
    _path = '/groups/{group_id}/variables'
    _obj_cls = GroupVariable
    _from_parent_attrs = {'group_id': 'id'}
    _create_attrs = RequiredOptional(required=('key', 'value'), optional=('protected', 'variable_type', 'masked'))
    _update_attrs = RequiredOptional(required=('key', 'value'), optional=('protected', 'variable_type', 'masked'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupVariable:
        return cast(GroupVariable, super().get(id=id, lazy=lazy, **kwargs))