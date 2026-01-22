from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
import requests
import gitlab
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, ListMixin, RefreshMixin, RetrieveMixin
from gitlab.types import RequiredOptional
from .discussions import ProjectCommitDiscussionManager  # noqa: F401
class ProjectCommitManager(RetrieveMixin, CreateMixin, RESTManager):
    _path = '/projects/{project_id}/repository/commits'
    _obj_cls = ProjectCommit
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('branch', 'commit_message', 'actions'), optional=('author_email', 'author_name'))
    _list_filters = ('all', 'ref_name', 'since', 'until', 'path', 'with_stats', 'first_parent', 'order', 'trailers')

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectCommit:
        return cast(ProjectCommit, super().get(id=id, lazy=lazy, **kwargs))