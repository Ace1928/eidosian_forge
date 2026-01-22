from __future__ import annotations
import asyncio
import functools
import hashlib
import io
import json
import os
import pathlib
import sys
import uuid
from typing import (
import bokeh
import js
import param
import pyodide # isort: split
from bokeh import __version__
from bokeh.core.serialization import Buffer, Serialized, Serializer
from bokeh.document import Document
from bokeh.document.json import PatchJson
from bokeh.embed.elements import script_for_render_items
from bokeh.embed.util import standalone_docs_json_and_render_items
from bokeh.embed.wrappers import wrap_in_script_tag
from bokeh.events import DocumentReady
from bokeh.io.doc import set_curdoc
from bokeh.model import Model
from bokeh.settings import settings as bk_settings
from bokeh.util.sampledata import (
from js import JSON, XMLHttpRequest
from ..config import config
from ..util import edit_readonly, isurl
from . import resources
from .document import MockSessionContext
from .loading import LOADING_INDICATOR_CSS_CLASS
from .mime_render import WriteCallbackStream, exec_with_return, format_mime
from .state import state
def _doc_json(doc: Document, root_els=None) -> Tuple[str, str, str]:
    """
    Serializes a Bokeh Document into JSON representations of the entire
    Document, the individual render_items and the ids of DOM nodes to
    render each item into.

    Arguments
    ---------
    doc: bokeh.document.Document
        The Bokeh document to serialize to JSON.
    root_els:
        A list of DOM nodes to render each root of the Document into.

    Returns
    -------
    docs_json: str
    render_items: str
    root_ids: str
    """
    docs_json, render_items = standalone_docs_json_and_render_items(doc.roots, suppress_callback_warning=True)
    render_items_json = [item.to_json() for item in render_items]
    root_ids = [m.id for m in doc.roots]
    if root_els:
        root_data = sorted([(el.getAttribute('data-root-id'), el.id) for el in root_els])
        render_items_json[0].update({'roots': {model_id: elid for (_, elid), model_id in zip(root_data, root_ids)}, 'root_ids': root_ids})
    return (json.dumps(docs_json), json.dumps(render_items_json), json.dumps(root_ids))