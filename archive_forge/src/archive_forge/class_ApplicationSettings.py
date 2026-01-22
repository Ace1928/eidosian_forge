from typing import Any, cast, Dict, Optional, Union
from gitlab import exceptions as exc
from gitlab import types
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import GetWithoutIdMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
class ApplicationSettings(SaveMixin, RESTObject):
    _id_attr = None