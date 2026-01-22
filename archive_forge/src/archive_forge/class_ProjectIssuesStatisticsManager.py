from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import GetWithoutIdMixin, RefreshMixin
from gitlab.types import ArrayAttribute
class ProjectIssuesStatisticsManager(GetWithoutIdMixin, RESTManager):
    _path = '/projects/{project_id}/issues_statistics'
    _obj_cls = ProjectIssuesStatistics
    _from_parent_attrs = {'project_id': 'id'}
    _list_filters = ('iids',)
    _types = {'iids': ArrayAttribute}

    def get(self, **kwargs: Any) -> ProjectIssuesStatistics:
        return cast(ProjectIssuesStatistics, super().get(**kwargs))