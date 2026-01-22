from typing import Any, cast, List, Optional, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectRunnerManager(CreateMixin, DeleteMixin, ListMixin, RESTManager):
    _path = '/projects/{project_id}/runners'
    _obj_cls = ProjectRunner
    _from_parent_attrs = {'project_id': 'id'}
    _create_attrs = RequiredOptional(required=('runner_id',))
    _list_filters = ('scope', 'tag_list')
    _types = {'tag_list': types.CommaSeparatedListAttribute}