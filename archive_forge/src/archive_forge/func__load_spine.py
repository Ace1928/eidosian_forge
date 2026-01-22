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
def _load_spine(self):
    spine = self.container.find('{%s}%s' % (NAMESPACES['OPF'], 'spine'))
    self.book.spine = [(t.get('idref'), t.get('linear', 'yes')) for t in spine]
    toc = spine.get('toc', '')
    self.book.set_direction(spine.get('page-progression-direction', None))
    nav_item = next((item for item in self.book.items if isinstance(item, EpubNav)), None)
    if toc:
        if not self.options.get('ignore_ncx') or not nav_item:
            try:
                ncxFile = self.read_file(zip_path.join(self.opf_dir, self.book.get_item_with_id(toc).get_name()))
            except KeyError:
                raise EpubException(-1, 'Can not find ncx file.')
            self._parse_ncx(ncxFile)