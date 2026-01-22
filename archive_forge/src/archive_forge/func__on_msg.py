from __future__ import annotations
import asyncio
import functools
import logging
import os
import sys
import threading
import traceback
import uuid
from typing import (
import param  # type: ignore
from bokeh.core.serialization import DeserializationError
from bokeh.document import Document
from bokeh.resources import Resources
from jinja2 import Template
from pyviz_comms import Comm  # type: ignore
from ._param import Align, Aspect, Margin
from .config import config, panel_extension
from .io import serve
from .io.document import create_doc_if_none_exists, init_doc
from .io.embed import embed_state
from .io.loading import start_loading_spinner, stop_loading_spinner
from .io.model import add_to_doc, patch_cds_msg
from .io.notebook import (
from .io.save import save
from .io.state import curdoc_locked, set_curdoc, state
from .util import escape, param_reprs
from .util.parameters import get_params_to_inherit
def _on_msg(self, ref: str, manager, msg) -> None:
    """
        Handles Protocol messages arriving from the client comm.
        """
    root, doc, comm = state._views[ref][1:]
    patch_cds_msg(root, msg)
    held = doc.callbacks.hold_value
    patch = manager.assemble(msg)
    doc.hold()
    try:
        patch.apply_to_document(doc, comm.id if comm else None)
    except DeserializationError:
        self.param.warning('Comm received message that could not be deserialized.')
    finally:
        doc.unhold()
        if held:
            doc.hold(held)