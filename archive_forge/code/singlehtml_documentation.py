from os import path
from typing import Any, Dict, List, Optional, Tuple, Union
from docutils import nodes
from docutils.nodes import Node
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.environment.adapters.toctree import TocTree
from sphinx.locale import __
from sphinx.util import logging, progress_message
from sphinx.util.console import darkgreen  # type: ignore
from sphinx.util.nodes import inline_all_toctrees

    A StandaloneHTMLBuilder subclass that puts the whole document tree on one
    HTML page.
    