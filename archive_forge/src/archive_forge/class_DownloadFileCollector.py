import os
from glob import glob
from os import path
from typing import Any, Dict, List, Set
from docutils import nodes
from docutils.nodes import Node
from docutils.utils import relative_path
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.locale import __
from sphinx.util import logging
from sphinx.util.i18n import get_image_filename_for_language, search_image_for_language
from sphinx.util.images import guess_mimetype
class DownloadFileCollector(EnvironmentCollector):
    """Download files collector for sphinx.environment."""

    def clear_doc(self, app: Sphinx, env: BuildEnvironment, docname: str) -> None:
        env.dlfiles.purge_doc(docname)

    def merge_other(self, app: Sphinx, env: BuildEnvironment, docnames: Set[str], other: BuildEnvironment) -> None:
        env.dlfiles.merge_other(docnames, other.dlfiles)

    def process_doc(self, app: Sphinx, doctree: nodes.document) -> None:
        """Process downloadable file paths. """
        for node in doctree.findall(addnodes.download_reference):
            targetname = node['reftarget']
            if '://' in targetname:
                node['refuri'] = targetname
            else:
                rel_filename, filename = app.env.relfn2path(targetname, app.env.docname)
                app.env.dependencies[app.env.docname].add(rel_filename)
                if not os.access(filename, os.R_OK):
                    logger.warning(__('download file not readable: %s') % filename, location=node, type='download', subtype='not_readable')
                    continue
                node['filename'] = app.env.dlfiles.add_file(app.env.docname, rel_filename)