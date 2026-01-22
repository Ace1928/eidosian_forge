from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import RequiredOptional
class ProjectProtectedBranchManager(NoUpdateMixin, RESTManager):
    _path = '/projects/{project_id}/protected_branches'
    _obj_cls = ProjectProtectedBranch
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('name',), optional=('push_access_level', 'merge_access_level', 'unprotect_access_level', 'allow_force_push', 'allowed_to_push', 'allowed_to_merge', 'allowed_to_unprotect', 'code_owner_approval_required'))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectProtectedBranch:
        return cast(ProjectProtectedBranch, super().get(id=id, lazy=lazy, **kwargs))