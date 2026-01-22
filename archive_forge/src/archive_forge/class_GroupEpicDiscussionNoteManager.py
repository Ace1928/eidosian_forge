from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import (  # noqa: F401
class GroupEpicDiscussionNoteManager(GetMixin, CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/groups/{group_id}/epics/{epic_id}/discussions/{discussion_id}/notes'
    _obj_cls = GroupEpicDiscussionNote
    _from_parent_attrs = {'group_id': 'group_id', 'epic_id': 'epic_id', 'discussion_id': 'id'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at',))
    _update_attrs = RequiredOptional(required=('body',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupEpicDiscussionNote:
        return cast(GroupEpicDiscussionNote, super().get(id=id, lazy=lazy, **kwargs))