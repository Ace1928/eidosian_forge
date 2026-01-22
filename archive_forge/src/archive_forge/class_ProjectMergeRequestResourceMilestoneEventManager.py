from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin, RetrieveMixin
class ProjectMergeRequestResourceMilestoneEventManager(RetrieveMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/resource_milestone_events'
    _obj_cls = ProjectMergeRequestResourceMilestoneEvent
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'iid'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMergeRequestResourceMilestoneEvent:
        return cast(ProjectMergeRequestResourceMilestoneEvent, super().get(id=id, lazy=lazy, **kwargs))