from __future__ import annotations
import pathlib
from typing import ClassVar, Dict, List
import param
from ...theme import Design
from ...theme.bootstrap import Bootstrap
from ..base import BasicTemplate, TemplateActions
class BootstrapTemplateActions(TemplateActions):
    _scripts: ClassVar[Dict[str, List[str] | str]] = {'render': "state.modal = new bootstrap.Modal(document.getElementById('pn-Modal'))", 'open_modal': 'state.modal.show()', 'close_modal': 'state.modal.hide()'}