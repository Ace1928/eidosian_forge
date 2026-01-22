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
def _download_sampledata(progress: bool=False) -> None:
    """
    Download bokeh sampledata
    """
    data_dir = external_data_dir(create=True)
    s3 = 'https://sampledata.bokeh.org'
    with open(pathlib.Path(_bk_util_dir).parent / 'sampledata.json') as f:
        files = json.load(f)
    for filename, md5 in files:
        real_name, ext = splitext(filename)
        if ext == '.zip':
            if not splitext(real_name)[1]:
                real_name += '.csv'
        else:
            real_name += ext
        real_path = data_dir / real_name
        if real_path.exists():
            with open(real_path, 'rb') as file:
                data = file.read()
            local_md5 = hashlib.md5(data).hexdigest()
            if local_md5 == md5:
                continue
        _download_file(s3, filename, data_dir, progress=progress)