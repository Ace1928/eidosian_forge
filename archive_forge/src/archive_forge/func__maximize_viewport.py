from __future__ import annotations
import logging # isort:skip
import io
import os
from contextlib import contextmanager
from os.path import abspath, expanduser, splitext
from tempfile import mkstemp
from typing import (
from ..core.types import PathLike
from ..document import Document
from ..embed import file_html
from ..resources import INLINE, Resources
from ..themes import Theme
from ..util.warnings import warn
from .state import State, curstate
from .util import default_filename
def _maximize_viewport(web_driver: WebDriver) -> tuple[int, int, int]:
    calculate_viewport_size = '        const root_view = Bokeh.index.roots[0]\n        const {width, height} = root_view.el.getBoundingClientRect()\n        return [Math.round(width), Math.round(height), window.devicePixelRatio]\n    '
    viewport_size: tuple[int, int, int] = web_driver.execute_script(calculate_viewport_size)
    calculate_window_size = '        const [width, height, dpr] = arguments\n        return [\n            // XXX: outer{Width,Height} can be 0 in headless mode under certain window managers\n            Math.round(Math.max(0, window.outerWidth - window.innerWidth) + width*dpr),\n            Math.round(Math.max(0, window.outerHeight - window.innerHeight) + height*dpr),\n        ]\n    '
    [width, height] = web_driver.execute_script(calculate_window_size, *viewport_size)
    eps = 100
    web_driver.set_window_size(width + eps, height + eps)
    return viewport_size