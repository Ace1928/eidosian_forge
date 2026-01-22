from typing import Any, cast, Union
from gitlab.base import RESTManager, RESTObject
from gitlab.mixins import CreateMixin, RetrieveMixin, SaveMixin, UpdateMixin
from gitlab.types import RequiredOptional
from .notes import (  # noqa: F401
class ProjectSnippetDiscussion(RESTObject):
    notes: ProjectSnippetDiscussionNoteManager