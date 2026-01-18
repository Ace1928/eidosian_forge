from typing import Any, Dict, List
from docutils import nodes
from docutils.nodes import Node
import sphinx
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import nested_parse_with_titles
from sphinx.util.typing import OptionSpec
Provides the ``ifconfig`` directive.

The ``ifconfig`` directive enables writing documentation
that is included depending on configuration variables.

Usage::

    .. ifconfig:: releaselevel in ('alpha', 'beta', 'rc')

       This stuff is only included in the built docs for unstable versions.

The argument for ``ifconfig`` is a plain Python expression, evaluated in the
namespace of the project configuration (that is, all variables from
``conf.py`` are available.)
