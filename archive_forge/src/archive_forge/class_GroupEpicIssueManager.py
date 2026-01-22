from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .events import GroupEpicResourceLabelEventManager  # noqa: F401
from .notes import GroupEpicNoteManager  # noqa: F401
class GroupEpicIssueManager(ListMixin, CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/groups/{group_id}/epics/{epic_iid}/issues'
    _obj_cls = GroupEpicIssue
    _from_parent_attrs = {'group_id': 'group_id', 'epic_iid': 'iid'}
    _create_attrs = RequiredOptional(required=('issue_id',))
    _update_attrs = RequiredOptional(optional=('move_before_id', 'move_after_id'))

    @exc.on_http_error(exc.GitlabCreateError)
    def create(self, data: Optional[Dict[str, Any]]=None, **kwargs: Any) -> GroupEpicIssue:
        """Create a new object.

        Args:
            data: Parameters to send to the server to create the
                         resource
            **kwargs: Extra options to send to the server (e.g. sudo)

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server cannot perform the request

        Returns:
            A new instance of the manage object class build with
                the data sent by the server
        """
        if TYPE_CHECKING:
            assert data is not None
        self._create_attrs.validate_attrs(data=data)
        path = f'{self.path}/{data.pop('issue_id')}'
        server_data = self.gitlab.http_post(path, **kwargs)
        if TYPE_CHECKING:
            assert isinstance(server_data, dict)
        server_data['epic_issue_id'] = server_data['id']
        return self._obj_cls(self, server_data)