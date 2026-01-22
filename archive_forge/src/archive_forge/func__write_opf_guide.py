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
def _write_opf_guide(self, root):
    if len(self.book.guide) > 0 and self.options.get('epub2_guide'):
        guide = etree.SubElement(root, 'guide', {})
        for item in self.book.guide:
            if 'item' in item:
                chap = item.get('item')
                if chap:
                    _href = chap.file_name
                    _title = chap.title
            else:
                _href = item.get('href', '')
                _title = item.get('title', '')
            if _title is None:
                _title = ''
            ref = etree.SubElement(guide, 'reference', {'type': item.get('type', ''), 'title': _title, 'href': _href})