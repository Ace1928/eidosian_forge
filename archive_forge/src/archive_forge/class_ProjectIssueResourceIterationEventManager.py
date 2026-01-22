from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin, RetrieveMixin
class ProjectIssueResourceIterationEventManager(RetrieveMixin, RESTManager):
    _path = '/projects/{project_id}/issues/{issue_iid}/resource_iteration_events'
    _obj_cls = ProjectIssueResourceIterationEvent
    _from_parent_attrs = {'project_id': 'project_id', 'issue_iid': 'iid'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectIssueResourceIterationEvent:
        return cast(ProjectIssueResourceIterationEvent, super().get(id=id, lazy=lazy, **kwargs))