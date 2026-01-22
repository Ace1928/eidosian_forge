import lxml
import os
import os.path as op
import sys
import re
import shutil
import tempfile
import zipfile
import codecs
from fnmatch import fnmatch
from itertools import islice
from lxml import etree
from pathlib import Path
from .uriutil import join_uri, translate_uri, uri_segment
from .uriutil import uri_last, uri_nextlast
from .uriutil import uri_parent, uri_grandparent
from .uriutil import uri_shape
from .uriutil import file_path
from .jsonutil import JsonTable, get_selection
from .pathutil import find_files, ensure_dir_exists
from .attributes import EAttrs
from .search import rpn_contraints, query_from_xml
from .errors import is_xnat_error, parse_put_error_message
from .errors import DataError, ProgrammingError, catch_error
from .provenance import Provenance
from .pipelines import Pipelines
from . import schema
from . import httputil
from . import downloadutils
from . import derivatives
import types
import pkgutil
import inspect
from urllib.parse import quote, unquote
class Assessor(EObject, metaclass=ElementType):

    def __init__(self, uri, interface):
        EObject.__init__(self, uri, interface)
        self.provenance = Provenance(self)

    def set_param(self, key, value):
        self.attrs.set('%s/parameters/addParam[name=%s]/addField' % (self.datatype(), key), value)

    def get_param(self, key):
        return self.xpath("//xnat:addParam[@name='%s']/child::text()" % key)[-1]

    def get_params(self):
        return self.xpath('//xnat:addParam/child::text()')[1::2]

    def params(self):
        return self.xpath('//xnat:addParam/attribute::*')