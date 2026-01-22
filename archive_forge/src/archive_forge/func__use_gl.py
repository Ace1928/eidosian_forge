from __future__ import annotations
import logging # isort:skip
import hashlib
import json
import os
from dataclasses import dataclass
from os.path import normpath
from pathlib import Path
from typing import (
from urllib.parse import urljoin
from ..core.has_props import HasProps
from ..core.templates import CSS_RESOURCES, JS_RESOURCES
from ..document.document import Document
from ..resources import Resources
from ..settings import settings
from ..util.compiler import bundle_models
from .util import contains_tex_string
def _use_gl(all_objs: set[HasProps]) -> bool:
    """ Whether a collection of Bokeh objects contains a plot requesting WebGL

    Args:
        objs (seq[HasProps or Document]) :

    Returns:
        bool

    """
    from ..models.plots import Plot
    return _any(all_objs, lambda obj: isinstance(obj, Plot) and obj.output_backend == 'webgl')