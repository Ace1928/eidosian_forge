from __future__ import annotations
import logging  # isort:skip
import json
import os
from os.path import (
from pathlib import PurePath
from typing import TypedDict
from sphinx.errors import SphinxError
from sphinx.util import ensuredir
from sphinx.util.display import status_iterator
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import GALLERY_DETAIL, GALLERY_PAGE
from .util import _REPO_TOP
class GalleryDetail(TypedDict):
    name: str
    path: str
    ref: str
    rst_file_path: str