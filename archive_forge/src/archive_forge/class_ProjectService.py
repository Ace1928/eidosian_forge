from typing import Any, cast, List, Union
from gitlab import cli
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
class ProjectService(ProjectIntegration):
    pass