from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, NoUpdateMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class ProjectHook(SaveMixin, ObjectDeleteMixin, RESTObject):
    _repr_attr = 'url'