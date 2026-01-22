from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, DownloadMixin, GetWithoutIdMixin, RefreshMixin
from gitlab.types import RequiredOptional
class GroupExportManager(GetWithoutIdMixin, CreateMixin, RESTManager):
    _path = '/groups/{group_id}/export'
    _obj_cls = GroupExport
    _from_parent_attrs = {'group_id': 'id'}

    def get(self, **kwargs: Any) -> GroupExport:
        return cast(GroupExport, super().get(**kwargs))