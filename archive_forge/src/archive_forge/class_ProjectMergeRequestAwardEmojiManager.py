from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import NoUpdateMixin, ObjectDeleteMixin
from gitlab.types import RequiredOptional
class ProjectMergeRequestAwardEmojiManager(NoUpdateMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/award_emoji'
    _obj_cls = ProjectMergeRequestAwardEmoji
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'iid'}
    _create_attrs = RequiredOptional(required=('name',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMergeRequestAwardEmoji:
        return cast(ProjectMergeRequestAwardEmoji, super().get(id=id, lazy=lazy, **kwargs))