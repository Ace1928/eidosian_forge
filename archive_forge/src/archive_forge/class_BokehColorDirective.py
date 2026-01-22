from __future__ import annotations
import logging  # isort:skip
from docutils import nodes
from docutils.parsers.rst.directives import unchanged
from bokeh.colors import named
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import COLOR_DETAIL
class BokehColorDirective(BokehDirective):
    has_content = False
    required_arguments = 1
    option_spec = {'module': unchanged}

    def run(self):
        color = self.arguments[0]
        html = COLOR_DETAIL.render(color=getattr(named, color).to_css(), text=color)
        node = nodes.raw('', html, format='html')
        return [node]