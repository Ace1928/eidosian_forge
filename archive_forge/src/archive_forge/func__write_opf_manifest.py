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
def _write_opf_manifest(self, root):
    manifest = etree.SubElement(root, 'manifest')
    _ncx_id = None
    for item in self.book.get_items():
        if not item.manifest:
            continue
        if isinstance(item, EpubNav):
            etree.SubElement(manifest, 'item', {'href': item.get_name(), 'id': item.id, 'media-type': item.media_type, 'properties': 'nav'})
        elif isinstance(item, EpubNcx):
            _ncx_id = item.id
            etree.SubElement(manifest, 'item', {'href': item.file_name, 'id': item.id, 'media-type': item.media_type})
        elif isinstance(item, EpubCover):
            etree.SubElement(manifest, 'item', {'href': item.file_name, 'id': item.id, 'media-type': item.media_type, 'properties': 'cover-image'})
        else:
            opts = {'href': item.file_name, 'id': item.id, 'media-type': item.media_type}
            if hasattr(item, 'properties') and len(item.properties) > 0:
                opts['properties'] = ' '.join(item.properties)
            if hasattr(item, 'media_overlay') and item.media_overlay is not None:
                opts['media-overlay'] = item.media_overlay
            if hasattr(item, 'media_duration') and item.media_duration is not None:
                opts['duration'] = item.media_duration
            etree.SubElement(manifest, 'item', opts)
    return _ncx_id