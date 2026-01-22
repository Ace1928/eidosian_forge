from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import (  # noqa: F401
class ProjectCommitDiscussionNoteManager(GetMixin, CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/repository/commits/{commit_id}/discussions/{discussion_id}/notes'
    _obj_cls = ProjectCommitDiscussionNote
    _from_parent_attrs = {'project_id': 'project_id', 'commit_id': 'commit_id', 'discussion_id': 'id'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at', 'position'))
    _update_attrs = RequiredOptional(required=('body',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectCommitDiscussionNote:
        return cast(ProjectCommitDiscussionNote, super().get(id=id, lazy=lazy, **kwargs))