from __future__ import annotations
import pathlib
from typing import (
import param
from bokeh.models import CustomJS
from ...config import config
from ...reactive import ReactiveHTML
from ..vanilla import VanillaTemplate
@param.depends('editable', watch=True, on_init=True)
def _add_editor(self) -> None:
    if not self.editable:
        return
    editor = TemplateEditor()
    editor.param.watch(self._sync_positions, 'layout')
    self._render_items['editor'] = (editor, [])