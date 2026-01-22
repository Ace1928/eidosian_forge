from typing import Any, cast, Dict, List, Optional, TYPE_CHECKING, Union
from gitlab import exceptions as exc
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class ProjectApprovalRule(SaveMixin, ObjectDeleteMixin, RESTObject):
    _id_attr = 'id'