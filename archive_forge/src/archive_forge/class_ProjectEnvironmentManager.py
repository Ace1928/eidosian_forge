from typing import Any, cast, Dict, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
class ProjectEnvironmentManager(RetrieveMixin, CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/environments'
    _obj_cls = ProjectEnvironment
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('name',), optional=('external_url',))
    _update_attrs = RequiredOptional(optional=('name', 'external_url'))
    _list_filters = ('name', 'search', 'states')

    def get(self, id: Union[str, int], lazy: bool=False, **kwargs: Any) -> ProjectEnvironment:
        return cast(ProjectEnvironment, super().get(id=id, lazy=lazy, **kwargs))