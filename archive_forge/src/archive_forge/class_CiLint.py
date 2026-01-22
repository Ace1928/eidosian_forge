from typing import Any, cast
from gitlab.base import RESTManager, RESTObject
from gitlab.cli import register_custom_action
from gitlab.exceptions import GitlabCiLintError
from gitlab.mixins import CreateMixin, GetWithoutIdMixin
from gitlab.types import RequiredOptional
class CiLint(RESTObject):
    _id_attr = None