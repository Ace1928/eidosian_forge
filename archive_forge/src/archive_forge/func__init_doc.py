from __future__ import annotations
import pathlib
from typing import (
import param
from bokeh.models import CustomJS
from ...config import config
from ...reactive import ReactiveHTML
from ..vanilla import VanillaTemplate
def _init_doc(self, doc: Optional[Document]=None, comm: Optional[Comm]=None, title: Optional[str]=None, notebook: bool=False, location: bool | Location=True):
    doc = super()._init_doc(doc, comm, title, notebook, location)
    doc.js_on_event('document_ready', CustomJS(code="\n          window.muuriGrid.getItems().map(item => scroll(item.getElement()));\n          for (const root of roots) {\n            root.sizing_mode = 'stretch_both';\n            if (root.children) {\n              for (const child of root) {\n                child.sizing_mode = 'stretch_both'\n              }\n            }\n          }\n          window.muuriGrid.refreshItems();\n          window.muuriGrid.layout();\n        ", args={'roots': [root for root in doc.roots if 'main' in root.tags]}))
    return doc