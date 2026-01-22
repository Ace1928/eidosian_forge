from typing import Any, Callable, cast, Dict, Iterator, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import utils
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import RefreshMixin, RetrieveMixin
from gitlab.types import ArrayAttribute
class ProjectJobManager(RetrieveMixin, RESTManager):
    _path = '/projects/{project_id}/jobs'
    _obj_cls = ProjectJob
    _from_parent_attrs = {'project_id': 'id'}
    _list_filters = ('scope',)
    _types = {'scope': ArrayAttribute}

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectJob:
        return cast(ProjectJob, super().get(id=id, lazy=lazy, **kwargs))