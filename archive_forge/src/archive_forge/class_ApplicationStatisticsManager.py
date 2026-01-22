from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import GetWithoutIdMixin, RefreshMixin
from gitlab.types import ArrayAttribute
class ApplicationStatisticsManager(GetWithoutIdMixin, RESTManager):
    _path = '/application/statistics'
    _obj_cls = ApplicationStatistics

    def get(self, **kwargs: Any) -> ApplicationStatistics:
        return cast(ApplicationStatistics, super().get(**kwargs))