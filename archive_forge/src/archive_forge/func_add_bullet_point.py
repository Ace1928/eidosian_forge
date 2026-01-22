from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
def add_bullet_point(app, fromdocname, docname, ref_name):
    line = nodes.line()
    line += nodes.Text('  • ', '  • ')
    newnode = nodes.reference('', '')
    innernode = nodes.emphasis(_(ref_name), _(ref_name))
    newnode['refdocname'] = docname
    newnode['refuri'] = app.builder.get_relative_uri(fromdocname, docname)
    newnode.append(innernode)
    line += newnode
    return line