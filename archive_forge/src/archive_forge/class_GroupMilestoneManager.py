from typing import Any, cast, TYPE_CHECKING, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject, RESTObjectList
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .issues import GroupIssue, GroupIssueManager, ProjectIssue, ProjectIssueManager
from .merge_requests import (
class GroupMilestoneManager(CRUDMixin, RESTManager):
    _path = '/groups/{group_id}/milestones'
    _obj_cls = GroupMilestone
    _from_parent_attrs = {'group_id': 'id'}
    _create_attrs = RequiredOptional(required=('title',), optional=('description', 'due_date', 'start_date'))
    _update_attrs = RequiredOptional(optional=('title', 'description', 'due_date', 'start_date', 'state_event'))
    _list_filters = ('iids', 'state', 'search')
    _types = {'iids': types.ArrayAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupMilestone:
        return cast(GroupMilestone, super().get(id=id, lazy=lazy, **kwargs))