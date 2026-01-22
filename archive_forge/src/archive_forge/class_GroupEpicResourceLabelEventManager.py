from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin, RetrieveMixin
class GroupEpicResourceLabelEventManager(RetrieveMixin, RESTManager):
    _path = '/groups/{group_id}/epics/{epic_id}/resource_label_events'
    _obj_cls = GroupEpicResourceLabelEvent
    _from_parent_attrs = {'group_id': 'group_id', 'epic_id': 'id'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupEpicResourceLabelEvent:
        return cast(GroupEpicResourceLabelEvent, super().get(id=id, lazy=lazy, **kwargs))