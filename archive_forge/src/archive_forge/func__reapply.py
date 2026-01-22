from __future__ import annotations
import functools
import os
import pathlib
from typing import (
import param
from bokeh.models import ImportedStyleSheet
from bokeh.themes import Theme as _BkTheme, _dark_minimal, built_in_themes
from ..config import config
from ..io.resources import (
from ..util import relative_to
def _reapply(self, viewable: Viewable, root: Model, old_models: List[Model]=None, isolated: bool=True, cache=None, document=None) -> None:
    ref = root.ref['id']
    for o in viewable.select():
        if o.design and (not isolated):
            continue
        elif not o.design and (not isolated):
            o._design = self
        if old_models and ref in o._models:
            if o._models[ref][0] in old_models:
                continue
        self._apply_modifiers(o, ref, self.theme, isolated, cache, document)