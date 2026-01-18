from __future__ import annotations
import html
import os
from os import path
from typing import Any
from docutils import nodes
from docutils.nodes import Element, Node, document
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.builders.html import StandaloneHTMLBuilder
from sphinx.config import Config
from sphinx.environment.adapters.indexentries import IndexEntries
from sphinx.locale import get_translation
from sphinx.util import logging
from sphinx.util.fileutil import copy_asset_file
from sphinx.util.nodes import NodeMatcher
from sphinx.util.osutil import make_filename_from_project, relpath
from sphinx.util.template import SphinxRenderer
def unknown_departure(self, node: Node) -> None:
    pass