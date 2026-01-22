from __future__ import annotations
from sphinx.util import logging  # isort:skip
from os.path import basename
from docutils import nodes
from sphinx.locale import _
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .util import get_sphinx_resources
class gallery_xrefs(nodes.General, nodes.Element):

    def __init__(self, *args, **kwargs):
        self.subfolder = kwargs.pop('subfolder', None)
        super().__init__(*args, **kwargs)