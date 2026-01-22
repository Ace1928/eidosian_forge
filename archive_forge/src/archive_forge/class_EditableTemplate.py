from __future__ import annotations
import pathlib
from typing import (
import param
from bokeh.models import CustomJS
from ...config import config
from ...reactive import ReactiveHTML
from ..vanilla import VanillaTemplate
class EditableTemplate(VanillaTemplate):
    """
    The `EditableTemplate` is a list based template with a header, sidebar, main and modal area.
    The template allow interactively dragging, resizing and hiding components on a grid.

    The template builds on top of Muuri and interact.js.

    Reference: https://panel.holoviz.org/reference/templates/EditableTemplate.html

    :Example:

    >>> pn.template.EditableTemplate(
    ...     site="Panel", title="EditableTemplate",
    ...     sidebar=[pn.pane.Markdown("## Settings"), some_slider],
    ...     main=[some_python_object]
    ... ).servable()
    """
    editable = param.Boolean(default=True, doc='\n      Whether the template layout should be editable.')
    layout = param.Dict(default={}, allow_refs=True, doc='\n      The layout definition of the template indexed by the id of\n      each component in the main area.')
    local_save = param.Boolean(default=True, doc='\n      Whether to enable saving to local storage.')
    _css = [pathlib.Path(__file__).parent.parent / 'vanilla' / 'vanilla.css', pathlib.Path(__file__).parent / 'editable.css']
    _resources: ClassVar[Dict[str, Dict[str, str]]] = {'css': {'lato': 'https://fonts.googleapis.com/css?family=Lato&subset=latin,latin-ext'}, 'js': {'interactjs': f'{config.npm_cdn}/interactjs@1.10.19/dist/interact.min.js', 'muuri': f'{config.npm_cdn}/muuri@0.9.5/dist/muuri.min.js', 'web-animation': f'{config.npm_cdn}/web-animations-js@2.3.2/web-animations.min.js'}}
    _template = pathlib.Path(__file__).parent / 'editable.html'

    def _update_vars(self):
        ids = {id(obj): next(iter(obj._models)) for obj in self.main}
        self._render_variables['layout'] = layout = {ids[iid]: dict(item, id=ids[iid]) for iid, item in self.layout.items()}
        self._render_variables['muuri_layout'] = list(layout.values())
        self._render_variables['editable'] = self.editable
        self._render_variables['local_save'] = self.local_save
        self._render_variables['loading_spinner'] = config.loading_spinner
        super()._update_vars()

    def _init_doc(self, doc: Optional[Document]=None, comm: Optional[Comm]=None, title: Optional[str]=None, notebook: bool=False, location: bool | Location=True):
        doc = super()._init_doc(doc, comm, title, notebook, location)
        doc.js_on_event('document_ready', CustomJS(code="\n          window.muuriGrid.getItems().map(item => scroll(item.getElement()));\n          for (const root of roots) {\n            root.sizing_mode = 'stretch_both';\n            if (root.children) {\n              for (const child of root) {\n                child.sizing_mode = 'stretch_both'\n              }\n            }\n          }\n          window.muuriGrid.refreshItems();\n          window.muuriGrid.layout();\n        ", args={'roots': [root for root in doc.roots if 'main' in root.tags]}))
        return doc

    @param.depends('editable', watch=True, on_init=True)
    def _add_editor(self) -> None:
        if not self.editable:
            return
        editor = TemplateEditor()
        editor.param.watch(self._sync_positions, 'layout')
        self._render_items['editor'] = (editor, [])

    def _sync_positions(self, event):
        ids = {mid: id(obj) for obj in self.main for mid in obj._models}
        self.layout.clear()
        self.layout.update({ids[item['id']]: item for item in event.new})
        self.param.trigger('layout')