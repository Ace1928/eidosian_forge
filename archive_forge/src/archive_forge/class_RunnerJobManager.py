from typing import Any, cast, List, Optional, Union
from gitlab import cli
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class RunnerJobManager(ListMixin, RESTManager):
    _path = '/runners/{runner_id}/jobs'
    _obj_cls = RunnerJob
    _from_parent_attrs = {'runner_id': 'id'}
    _list_filters = ('status',)