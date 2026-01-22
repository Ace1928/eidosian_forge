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
class ImageCollector(EnvironmentCollector):
    """Image files collector for sphinx.environment."""

    def clear_doc(self, app: Sphinx, env: BuildEnvironment, docname: str) -> None:
        env.images.purge_doc(docname)

    def merge_other(self, app: Sphinx, env: BuildEnvironment, docnames: Set[str], other: BuildEnvironment) -> None:
        env.images.merge_other(docnames, other.images)

    def process_doc(self, app: Sphinx, doctree: nodes.document) -> None:
        """Process and rewrite image URIs."""
        docname = app.env.docname
        for node in doctree.findall(nodes.image):
            candidates: Dict[str, str] = {}
            node['candidates'] = candidates
            imguri = node['uri']
            if imguri.startswith('data:'):
                candidates['?'] = imguri
                continue
            elif imguri.find('://') != -1:
                candidates['?'] = imguri
                continue
            if imguri.endswith(os.extsep + '*'):
                rel_imgpath, full_imgpath = app.env.relfn2path(imguri, docname)
                node['uri'] = rel_imgpath
                i18n_imguri = get_image_filename_for_language(imguri, app.env)
                _, full_i18n_imgpath = app.env.relfn2path(i18n_imguri, docname)
                self.collect_candidates(app.env, full_i18n_imgpath, candidates, node)
                self.collect_candidates(app.env, full_imgpath, candidates, node)
            else:
                imguri = search_image_for_language(imguri, app.env)
                node['uri'], _ = app.env.relfn2path(imguri, docname)
                candidates['*'] = node['uri']
            for imgpath in candidates.values():
                app.env.dependencies[docname].add(imgpath)
                if not os.access(path.join(app.srcdir, imgpath), os.R_OK):
                    logger.warning(__('image file not readable: %s') % imgpath, location=node, type='image', subtype='not_readable')
                    continue
                app.env.images.add_file(docname, imgpath)

    def collect_candidates(self, env: BuildEnvironment, imgpath: str, candidates: Dict[str, str], node: Node) -> None:
        globbed: Dict[str, List[str]] = {}
        for filename in glob(imgpath):
            new_imgpath = relative_path(path.join(env.srcdir, 'dummy'), filename)
            try:
                mimetype = guess_mimetype(filename)
                if mimetype is None:
                    basename, suffix = path.splitext(filename)
                    mimetype = 'image/x-' + suffix[1:]
                if mimetype not in candidates:
                    globbed.setdefault(mimetype, []).append(new_imgpath)
            except OSError as err:
                logger.warning(__('image file %s not readable: %s') % (filename, err), location=node, type='image', subtype='not_readable')
        for key, files in globbed.items():
            candidates[key] = sorted(files, key=len)[0]