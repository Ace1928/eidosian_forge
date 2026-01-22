from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import ArrayAttribute, RequiredOptional
class GroupAccessToken(ObjectDeleteMixin, ObjectRotateMixin, RESTObject):
    pass