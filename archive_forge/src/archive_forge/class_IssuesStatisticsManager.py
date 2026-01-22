from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import GetWithoutIdMixin, RefreshMixin
from gitlab.types import ArrayAttribute
class IssuesStatisticsManager(GetWithoutIdMixin, RESTManager):
    _path = '/issues_statistics'
    _obj_cls = IssuesStatistics
    _list_filters = ('iids',)
    _types = {'iids': ArrayAttribute}

    def get(self, **kwargs: Any) -> IssuesStatistics:
        return cast(IssuesStatistics, super().get(**kwargs))