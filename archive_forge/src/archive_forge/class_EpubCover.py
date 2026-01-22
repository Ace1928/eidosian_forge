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
class EpubCover(EpubItem):
    """
    Represents Cover image in the EPUB file.
    """

    def __init__(self, uid='cover-img', file_name=''):
        super(EpubCover, self).__init__(uid=uid, file_name=file_name)

    def get_type(self):
        return ebooklib.ITEM_COVER

    def __str__(self):
        return '<EpubCover:%s:%s>' % (self.id, self.file_name)