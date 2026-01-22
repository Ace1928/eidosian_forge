from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, ListMixin, RefreshMixin, RetrieveMixin
from gitlab.types import RequiredOptional
from .discussions import ProjectCommitDiscussionManager  # noqa: F401
class ProjectCommitStatusManager(ListMixin, CreateMixin, RESTManager):
    _path = '/projects/{project_id}/repository/commits/{commit_id}/statuses'
    _obj_cls = ProjectCommitStatus
    _from_parent_attrs = {'project_id': 'project_id', 'commit_id': 'id'}
    _create_attrs = RequiredOptional(required=('state',), optional=('description', 'name', 'context', 'ref', 'target_url', 'coverage'))

    @exc.on_http_error(exc.GitlabCreateError)
    def create(self, data: Optional[Dict[str, Any]]=None, **kwargs: Any) -> ProjectCommitStatus:
        """Create a new object.

        Args:
            data: Parameters to send to the server to create the
                         resource
            **kwargs: Extra options to send to the server (e.g. sudo or
                      'ref_name', 'stage', 'name', 'all')

        Raises:
            GitlabAuthenticationError: If authentication is not correct
            GitlabCreateError: If the server cannot perform the request

        Returns:
            A new instance of the manage object class build with
                the data sent by the server
        """
        base_path = '/projects/{project_id}/statuses/{commit_id}'
        path: Optional[str]
        if data is not None and 'project_id' in data and ('commit_id' in data):
            path = base_path.format(**data)
        else:
            path = self._compute_path(base_path)
        if TYPE_CHECKING:
            assert path is not None
        return cast(ProjectCommitStatus, CreateMixin.create(self, data, path=path, **kwargs))