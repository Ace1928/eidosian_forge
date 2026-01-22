import os
import re
from math import ceil
from typing import Any, Dict, List, Optional, Tuple
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.locale import __
from sphinx.transforms import SphinxTransform
from sphinx.util import epoch_to_rfc1123, logging, requests, rfc1123_to_epoch, sha1
from sphinx.util.images import get_image_extension, guess_mimetype, parse_data_uri
from sphinx.util.osutil import ensuredir
class DataURIExtractor(BaseImageConverter):
    default_priority = 150

    def match(self, node: nodes.image) -> bool:
        if self.app.builder.supported_remote_images == []:
            return False
        elif self.app.builder.supported_data_uri_images is True:
            return False
        else:
            return node['uri'].startswith('data:')

    def handle(self, node: nodes.image) -> None:
        image = parse_data_uri(node['uri'])
        ext = get_image_extension(image.mimetype)
        if ext is None:
            logger.warning(__('Unknown image format: %s...'), node['uri'][:32], location=node)
            return
        ensuredir(os.path.join(self.imagedir, 'embeded'))
        digest = sha1(image.data).hexdigest()
        path = os.path.join(self.imagedir, 'embeded', digest + ext)
        self.app.env.original_image_uri[path] = node['uri']
        with open(path, 'wb') as f:
            f.write(image.data)
        node['candidates'].pop('?')
        node['candidates'][image.mimetype] = path
        node['uri'] = path
        self.app.env.images.add_file(self.env.docname, path)