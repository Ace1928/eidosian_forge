from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectMergeRequestApprovalStateManager(GetWithoutIdMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/approval_state'
    _obj_cls = ProjectMergeRequestApprovalState
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'iid'}

    def get(self, **kwargs: Any) -> ProjectMergeRequestApprovalState:
        return cast(ProjectMergeRequestApprovalState, super().get(**kwargs))