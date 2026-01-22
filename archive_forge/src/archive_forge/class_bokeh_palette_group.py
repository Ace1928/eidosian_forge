from __future__ import annotations
import logging  # isort:skip
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.errors import SphinxError
import bokeh.palettes as bp
from . import PARALLEL_SAFE
from .templates import PALETTE_GROUP_DETAIL
class bokeh_palette_group(nodes.General, nodes.Element):

    @staticmethod
    def visit_html(visitor, node):
        visitor.body.append('<div class="container-fluid"><div class="row">')
        group = getattr(bp, node['group'], None)
        if not isinstance(group, dict):
            group_name = node['group']
            raise SphinxError(f'invalid palette group name {group_name}')
        names = sorted(group)
        for name in names:
            palettes = group[name]
            numbers = [x for x in sorted(palettes) if x < 30]
            html = PALETTE_GROUP_DETAIL.render(name=name, numbers=numbers, palettes=palettes)
            visitor.body.append(html)
        visitor.body.append('</div></div>')
        raise nodes.SkipNode
    html = (visit_html.__func__, None)