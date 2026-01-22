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
def assemble_toc_fignumbers(self) -> Dict[str, Dict[str, Dict[str, Tuple[int, ...]]]]:
    new_fignumbers: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    for docname, fignumlist in self.env.toc_fignumbers.items():
        for figtype, fignums in fignumlist.items():
            alias = '%s/%s' % (docname, figtype)
            new_fignumbers.setdefault(alias, {})
            for id, fignum in fignums.items():
                new_fignumbers[alias][id] = fignum
    return {self.config.root_doc: new_fignumbers}