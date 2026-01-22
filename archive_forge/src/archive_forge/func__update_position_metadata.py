from __future__ import annotations
import ast
import html
import json
import logging
import os
import pathlib
import re
import sys
import traceback
import urllib.parse as urlparse
from contextlib import contextmanager
from types import ModuleType
from typing import IO, Any, Callable
import bokeh.command.util
from bokeh.application.handlers.code import CodeHandler
from bokeh.application.handlers.code_runner import CodeRunner
from bokeh.application.handlers.handler import Handler, handle_exception
from bokeh.core.types import PathLike
from bokeh.document import Document
from bokeh.io.doc import curdoc, patch_curdoc, set_curdoc as bk_set_curdoc
from bokeh.util.dependencies import import_required
from ..config import config
from .mime_render import MIME_RENDERERS
from .profile import profile_ctx
from .reload import record_modules
from .state import state
def _update_position_metadata(self, event):
    """
        Maps EditableTemplate update events to cells in the original
        notebook and then overwrites notebook metadata with updated
        layout information.
        """
    nb = self._nb
    doc = event.obj._documents[-1]
    outputs = state._session_outputs[doc]
    cell_data, cell_ids = ({}, {})
    for cell in nb['cells']:
        if cell['id'] in outputs:
            out = outputs[cell['id']]
            cell_ids[id(out)] = cell['id']
            spec = dict(event.new[id(out)])
            del spec['id']
            cell_data[cell['id']] = spec
    order = [cell_ids[obj_id] for obj_id in event.new]
    nb_layout = {'cells': cell_data, 'order': order}
    nb_path = pathlib.Path(self._runner.path)
    path = nb_path.parent / f'.{nb_path.name}.layout'
    with open(path, 'w') as f:
        json.dump(nb_layout, f)
    self._stale = True