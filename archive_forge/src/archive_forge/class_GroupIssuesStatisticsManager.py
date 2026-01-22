from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import GetWithoutIdMixin, RefreshMixin
from gitlab.types import ArrayAttribute
class GroupIssuesStatisticsManager(GetWithoutIdMixin, RESTManager):
    _path = '/groups/{group_id}/issues_statistics'
    _obj_cls = GroupIssuesStatistics
    _from_parent_attrs = {'group_id': 'id'}
    _list_filters = ('iids',)
    _types = {'iids': ArrayAttribute}

    def get(self, **kwargs: Any) -> GroupIssuesStatistics:
        return cast(GroupIssuesStatistics, super().get(**kwargs))