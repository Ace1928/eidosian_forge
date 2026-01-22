from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.exceptions import GitlabInvitationError
from gitlab.mixins import CRUDMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import ArrayAttribute, CommaSeparatedListAttribute, RequiredOptional
class GroupInvitationManager(InvitationMixin, RESTManager):
    _path = '/groups/{group_id}/invitations'
    _obj_cls = GroupInvitation
    _from_parent_attrs = {'group_id': 'id'}
    _create_attrs = RequiredOptional(required=('access_level',), optional=('expires_at', 'invite_source', 'tasks_to_be_done', 'tasks_project_id'), exclusive=('email', 'user_id'))
    _update_attrs = RequiredOptional(optional=('access_level', 'expires_at'))
    _list_filters = ('query',)
    _types = {'email': CommaSeparatedListAttribute, 'user_id': CommaSeparatedListAttribute, 'tasks_to_be_done': ArrayAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> GroupInvitation:
        return cast(GroupInvitation, super().get(id=id, lazy=lazy, **kwargs))