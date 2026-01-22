from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectApprovalManager(GetWithoutIdMixin, UpdateMixin, RESTManager):
    _path = '/projects/{project_id}/approvals'
    _obj_cls = ProjectApproval
    _from_parent_attrs = {'project_id': 'id'}
    _update_attrs = RequiredOptional(optional=('approvals_before_merge', 'reset_approvals_on_push', 'disable_overriding_approvers_per_merge_request', 'merge_requests_author_approval', 'merge_requests_disable_committers_approval'))
    _update_method = UpdateMethod.POST

    def get(self, **kwargs: Any) -> ProjectApproval:
        return cast(ProjectApproval, super().get(**kwargs))