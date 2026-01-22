from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import (
from gitlab.types import RequiredOptional
from .award_emojis import (  # noqa: F401
class ProjectMergeRequestDiscussionNote(SaveMixin, ObjectDeleteMixin, RESTObject):
    pass