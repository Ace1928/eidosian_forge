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
class EpubSMIL(EpubItem):

    def __init__(self, uid=None, file_name='', content=None):
        super(EpubSMIL, self).__init__(uid=uid, file_name=file_name, media_type='application/smil+xml', content=content)

    def get_type(self):
        return ebooklib.ITEM_SMIL

    def __str__(self):
        return '<EpubSMIL:%s:%s>' % (self.id, self.file_name)