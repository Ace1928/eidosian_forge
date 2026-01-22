from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CRUDMixin, NoUpdateMixin, ObjectDeleteMixin, SaveMixin
from gitlab.types import RequiredOptional
class GroupHook(SaveMixin, ObjectDeleteMixin, RESTObject):
    _repr_attr = 'url'