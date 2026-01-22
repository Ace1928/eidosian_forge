from __future__ import annotations
import logging # isort:skip
from typing import (
from .. import __version__
from ..core.templates import (
from ..document.document import DEFAULT_TITLE, Document
from ..model import Model
from ..resources import Resources, ResourcesLike
from ..themes import Theme
from .bundle import Script, bundle_for_objs_and_resources
from .elements import html_page_for_render_items, script_for_render_items
from .util import (
from .wrappers import wrap_in_onload
class StandaloneEmbedJson(TypedDict):
    target_id: ID | None
    root_id: ID
    doc: DocJson
    version: str