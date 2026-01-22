from __future__ import annotations
from base64 import b64encode
from pathlib import Path
from typing import (
import param
from param.parameterized import eval_function_with_deps, iscoroutinefunction
from pyviz_comms import JupyterComm
from ..io.notebook import push
from ..io.resources import CDN_DIST
from ..io.state import state
from ..models import (
from ..util import lazy_load
from .base import Widget
from .button import BUTTON_STYLES, BUTTON_TYPES, IconMixin
from .indicators import Progress  # noqa
@param.depends('embed', 'file', 'callback', watch=True)
def _update_embed(self):
    if self.embed:
        self._transfer()