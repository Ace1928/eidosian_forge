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
@progress_message(__('writing additional files'))
def write_additional_files(self) -> None:
    for pagename, template in self.config.html_additional_pages.items():
        logger.info(' ' + pagename, nonl=True)
        self.handle_page(pagename, {}, template)
    if self.config.html_use_opensearch:
        logger.info(' opensearch', nonl=True)
        fn = path.join(self.outdir, '_static', 'opensearch.xml')
        self.handle_page('opensearch', {}, 'opensearch.xml', outfilename=fn)