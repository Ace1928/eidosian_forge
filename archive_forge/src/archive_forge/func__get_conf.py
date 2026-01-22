from __future__ import annotations
import html
import json
import os
import typing as t
import uuid
import warnings
from pathlib import Path
from jinja2 import (
from jupyter_core.paths import jupyter_path
from nbformat import NotebookNode
from traitlets import Bool, Dict, HasTraits, List, Unicode, default, observe, validate
from traitlets.config import Config
from traitlets.utils.importstring import import_item
from nbconvert import filters
from .exporter import Exporter
def _get_conf(self):
    conf: dict[str, t.Any] = {}
    for path in map(Path, self.template_paths):
        conf_path = path / 'conf.json'
        if conf_path.exists():
            with conf_path.open() as f:
                conf = recursive_update(conf, json.load(f))
    return conf