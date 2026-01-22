from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import (  # noqa: F401
class ProjectSnippetDiscussionNoteManager(GetMixin, CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/snippets/{snippet_id}/discussions/{discussion_id}/notes'
    _obj_cls = ProjectSnippetDiscussionNote
    _from_parent_attrs = {'project_id': 'project_id', 'snippet_id': 'snippet_id', 'discussion_id': 'id'}
    _create_attrs = RequiredOptional(required=('body',), optional=('created_at',))
    _update_attrs = RequiredOptional(required=('body',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectSnippetDiscussionNote:
        return cast(ProjectSnippetDiscussionNote, super().get(id=id, lazy=lazy, **kwargs))