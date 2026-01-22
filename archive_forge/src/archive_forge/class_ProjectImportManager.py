from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, DownloadMixin, GetWithoutIdMixin, RefreshMixin
from gitlab.types import RequiredOptional
class ProjectImportManager(GetWithoutIdMixin, RESTManager):
    _path = '/projects/{project_id}/import'
    _obj_cls = ProjectImport
    _from_parent_attrs = {'project_id': 'id'}

    def get(self, **kwargs: Any) -> ProjectImport:
        return cast(ProjectImport, super().get(**kwargs))