from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin, RetrieveMixin
class ProjectMergeRequestResourceStateEventManager(RetrieveMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/resource_state_events'
    _obj_cls = ProjectMergeRequestResourceStateEvent
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'iid'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMergeRequestResourceStateEvent:
        return cast(ProjectMergeRequestResourceStateEvent, super().get(id=id, lazy=lazy, **kwargs))