from __future__ import annotations
import re
import sys
from typing import (
import param  # type: ignore
from pyviz_comms import Comm, JupyterComm  # type: ignore
from ..io.resources import CDN_DIST
from ..util import lazy_load
from .base import ModelPane
def _get_model_type(self, root: Model, comm: Comm | None) -> Type[Model]:
    module = self.renderer
    if module is None:
        if 'panel.models.mathjax' in sys.modules and 'panel.models.katex' not in sys.modules:
            module = 'mathjax'
        else:
            module = 'katex'
    model = 'KaTeX' if module == 'katex' else 'MathJax'
    return lazy_load(f'panel.models.{module}', model, isinstance(comm, JupyterComm), root)