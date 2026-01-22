from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, RetrieveMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
from .notes import (  # noqa: F401
class ProjectMergeRequestDiscussionManager(RetrieveMixin, CreateMixin, UpdateMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/discussions'
    _obj_cls = ProjectMergeRequestDiscussion
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'iid'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at', 'position'))
    _update_attrs = RequiredOptional(required=('resolved',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMergeRequestDiscussion:
        return cast(ProjectMergeRequestDiscussion, super().get(id=id, lazy=lazy, **kwargs))