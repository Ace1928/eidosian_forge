from typing import Any, cast, Union
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
class DeployTokenManager(ListMixin, RESTManager):
    _path = '/deploy_tokens'
    _obj_cls = DeployToken