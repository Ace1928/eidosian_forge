import zipfile
import six
import logging
import uuid
import warnings
import posixpath as zip_path
import os.path
from collections import OrderedDict
from lxml import etree
import ebooklib
from ebooklib.utils import parse_string, parse_html_string, guess_type, get_pages_for_items
def _load_container(self):
    meta_inf = self.read_file('META-INF/container.xml')
    tree = parse_string(meta_inf)
    for root_file in tree.findall('//xmlns:rootfile[@media-type]', namespaces={'xmlns': NAMESPACES['CONTAINERNS']}):
        if root_file.get('media-type') == 'application/oebps-package+xml':
            self.opf_file = root_file.get('full-path')
            self.opf_dir = zip_path.dirname(self.opf_file)