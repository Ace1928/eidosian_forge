from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import (  # noqa: F401
class ProjectMergeRequestNoteManager(CRUDMixin, RESTManager):
    _path = '/projects/{project_id}/merge_requests/{mr_iid}/notes'
    _obj_cls = ProjectMergeRequestNote
    _from_parent_attrs = {'project_id': 'project_id', 'mr_iid': 'iid'}
    _create_attrs = RequiredOptional(required=('body',))
    _update_attrs = RequiredOptional(required=('body',))

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectMergeRequestNote:
        return cast(ProjectMergeRequestNote, super().get(id=id, lazy=lazy, **kwargs))