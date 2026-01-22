from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import (  # noqa: F401
class ProjectMergeRequestDiscussionNoteManager(GetMixin, CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/discussions/{discussion_id}/notes'
    _obj_cls = ProjectMergeRequestDiscussionNote
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'mr_iid', 'discussion_id': 'id'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at',))
    _update_attrs = RequiredOptional(required=('body',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMergeRequestDiscussionNote:
        return cast(ProjectMergeRequestDiscussionNote, super().get(id=id, lazy=lazy, **kwargs))