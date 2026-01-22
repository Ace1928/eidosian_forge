from typing import Any, cast, Dict, Optional, TYPE_CHECKING, Union
import requests
from gitlab import cli
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectPipelineScheduleVariableManager(CreateMixin, UpdateMixin, DeleteMixin, RESTManager):
    _path = '/projects/{project_id}/pipeline_schedules/{pipeline_schedule_id}/variables'
    _obj_cls = ProjectPipelineScheduleVariable
    _from_parent_attrs = {'project_id': 'project_id', 'pipeline_schedule_id': 'id'}
    _create_attrs = RequiredOptional(required=('key', 'value'))
    _update_attrs = RequiredOptional(required=('key', 'value'))