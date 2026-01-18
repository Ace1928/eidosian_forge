import hashlib
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.errors import ConfigError, ExtensionError
import matplotlib as mpl
from matplotlib import _api, mathtext
from matplotlib.rcsetup import validate_float_or_None
def visit_latex_math_latex(self, node):
    inline = isinstance(node.parent, nodes.TextElement)
    if inline:
        self.body.append('$%s$' % node['latex'])
    else:
        self.body.extend(['\\begin{equation}', node['latex'], '\\end{equation}'])