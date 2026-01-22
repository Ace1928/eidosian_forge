from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import RequiredOptional
class ProjectSnippetAwardEmojiManager(NoUpdateMixin, RESTManager):
    _path = '/projects/{project_id}/snippets/{snippet_id}/award_emoji'
    _obj_cls = ProjectSnippetAwardEmoji
    _from_parent_attrs = {'project_id': 'project_id', 'snippet_id': 'id'}
    _create_attrs = RequiredOptional(required=('name',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectSnippetAwardEmoji:
        return cast(ProjectSnippetAwardEmoji, super().get(id=id, lazy=lazy, **kwargs))