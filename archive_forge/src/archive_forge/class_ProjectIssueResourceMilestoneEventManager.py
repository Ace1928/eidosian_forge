from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import ListMixin, RetrieveMixin
class ProjectIssueResourceMilestoneEventManager(RetrieveMixin, RESTManager):
    _path = '/projects/{project_id}/issues/{issue_iid}/resource_milestone_events'
    _obj_cls = ProjectIssueResourceMilestoneEvent
    _from_parent_attrs = {'project_id': 'project_id', 'issue_iid': 'iid'}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectIssueResourceMilestoneEvent:
        return cast(ProjectIssueResourceMilestoneEvent, super().get(id=id, lazy=lazy, **kwargs))