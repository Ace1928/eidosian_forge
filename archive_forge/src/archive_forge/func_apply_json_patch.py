from __future__ import annotations
import logging # isort:skip
import gc
import weakref
from json import loads
from typing import TYPE_CHECKING, Any, Iterable
from jinja2 import Template
from ..core.enums import HoldPolicyType
from ..core.has_props import is_DataModel
from ..core.query import find, is_single_string_selector
from ..core.serialization import (
from ..core.templates import FILE
from ..core.types import ID
from ..core.validation import check_integrity, process_validation_issues
from ..events import Event
from ..model import Model
from ..themes import Theme, built_in_themes, default as default_theme
from ..util.serialization import make_id
from ..util.strings import nice_join
from ..util.version import __version__
from .callbacks import (
from .events import (
from .json import DocJson, PatchJson
from .models import DocumentModelManager
from .modules import DocumentModuleManager
def apply_json_patch(self, patch_json: PatchJson | Serialized[PatchJson], *, setter: Setter | None=None) -> None:
    """ Apply a JSON patch object and process any resulting events.

        Args:
            patch (JSON-data) :
                The JSON-object containing the patch to apply.

            setter (ClientSession or ServerSession or None, optional) :
                This is used to prevent "boomerang" updates to Bokeh apps.
                (default: None)

                In the context of a Bokeh server application, incoming updates
                to properties will be annotated with the session that is
                doing the updating. This value is propagated through any
                subsequent change notifications that the update triggers.
                The session can compare the event setter to itself, and
                suppress any updates that originate from itself.

        Returns:
            None

        """
    deserializer = Deserializer(list(self.models), setter=setter)
    try:
        patch: PatchJson = deserializer.deserialize(patch_json)
    except UnknownReferenceError as error:
        if self.models.seen(error.id):
            logging.warning(f'Dropping a patch because it contains a previously known reference (id={error.id!r}). Most of the time this is harmless and usually a result of updating a model on one side of a communications channel while it was being removed on the other end.')
            return
        else:
            raise
    events = patch['events']
    assert isinstance(events, list)
    for event in events:
        DocumentPatchedEvent.handle_event(self, event, setter)
    self.models.flush_synced(lambda model: not deserializer.has_ref(model))