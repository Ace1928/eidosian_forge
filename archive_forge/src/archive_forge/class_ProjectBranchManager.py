from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import RequiredOptional
class ProjectBranchManager(NoUpdateMixin, RESTManager):
    _path = '/projects/{project_id}/repository/branches'
    _obj_cls = ProjectBranch
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('branch', 'ref'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectBranch:
        return cast(ProjectBranch, super().get(id=id, lazy=lazy, **kwargs))