from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class GroupBoardListManager(CRUDMixin, RESTManager):
    _path = '/groups/{group_id}/boards/{board_id}/lists'
    _obj_cls = GroupBoardList
    _from_parent_attrs = {'group_id': 'group_id', 'board_id': 'id'}
    _create_attrs = RequiredOptional(exclusive=('label_id', 'assignee_id', 'milestone_id'))
    _update_attrs = RequiredOptional(required=('position',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupBoardList:
        return cast(GroupBoardList, super().get(id=id, lazy=lazy, **kwargs))