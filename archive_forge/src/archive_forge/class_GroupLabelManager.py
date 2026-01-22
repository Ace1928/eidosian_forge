from typing import Any, cast, Dict, Optional, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class GroupLabelManager(RetrieveMixin, CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/groups/{group_id}/labels'
    _obj_cls = GroupLabel
    _from_parent_attrs = {'group_id': 'id'}
    _create_attrs = RequiredOptional(required=('name', 'color'), optional=('description', 'priority'))
    _update_attrs = RequiredOptional(required=('name',), optional=('new_name', 'color', 'description', 'priority'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupLabel:
        return cast(GroupLabel, super().get(id=id, lazy=lazy, **kwargs))

    def update(self, name: Optional[str], new_data: Optional[Dict[str, Any]]=None, **kwargs: Any) -> Dict[str, Any]:
        """Update a Label on the server.

        Args:
            name: The name of the label
            **kwargs: Extra options to send to the server (e.g. sudo)
        """
        new_data = new_data or {}
        if name:
            new_data['name'] = name
        return super().update(id=None, new_data=new_data, **kwargs)