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
def _write_opf(self):
    package_attributes = {'xmlns': NAMESPACES['OPF'], 'unique-identifier': self.book.IDENTIFIER_ID, 'version': '3.0'}
    if self.book.direction and self.options['package_direction']:
        package_attributes['dir'] = self.book.direction
    root = etree.Element('package', package_attributes)
    prefixes = ['rendition: http://www.idpf.org/vocab/rendition/#'] + self.book.prefixes
    root.attrib['prefix'] = ' '.join(prefixes)
    self._write_opf_metadata(root)
    _ncx_id = self._write_opf_manifest(root)
    self._write_opf_spine(root, _ncx_id)
    self._write_opf_guide(root)
    self._write_opf_bindings(root)
    self._write_opf_file(root)