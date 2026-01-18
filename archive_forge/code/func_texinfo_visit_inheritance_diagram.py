import builtins
import inspect
import re
from importlib import import_module
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.ext.graphviz import (figure_wrapper, graphviz, render_dot_html, render_dot_latex,
from sphinx.util import md5
from sphinx.util.docutils import SphinxDirective
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
from sphinx.writers.latex import LaTeXTranslator
from sphinx.writers.texinfo import TexinfoTranslator
def texinfo_visit_inheritance_diagram(self: TexinfoTranslator, node: inheritance_diagram) -> None:
    """
    Output the graph for Texinfo.  This will insert a PNG.
    """
    graph = node['graph']
    graph_hash = get_graph_hash(node)
    name = 'inheritance%s' % graph_hash
    dotcode = graph.generate_dot(name, env=self.builder.env, graph_attrs={'size': '"6.0,6.0"'})
    render_dot_texinfo(self, node, dotcode, {}, 'inheritance')
    raise nodes.SkipNode