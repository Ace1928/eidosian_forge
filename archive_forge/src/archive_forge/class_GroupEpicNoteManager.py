from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import (  # noqa: F401
class GroupEpicNoteManager(CRUDMixin, RESTManager):
    _path = '/groups/{group_id}/epics/{epic_id}/notes'
    _obj_cls = GroupEpicNote
    _from_parent_attrs = {'group_id': 'group_id', 'epic_id': 'id'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at',))
    _update_attrs = RequiredOptional(required=('body',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupEpicNote:
        return cast(GroupEpicNote, super().get(id=id, lazy=lazy, **kwargs))