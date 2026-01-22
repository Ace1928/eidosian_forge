from __future__ import annotations
import datetime
import fnmatch
import logging
import optparse
import os
import re
import shutil
import sys
import tempfile
from collections import OrderedDict
from configparser import RawConfigParser
from io import StringIO
from typing import Iterable
from babel import Locale, localedata
from babel import __version__ as VERSION
from babel.core import UnknownLocaleError
from babel.messages.catalog import DEFAULT_HEADER, Catalog
from babel.messages.extract import (
from babel.messages.mofile import write_mo
from babel.messages.pofile import read_po, write_po
from babel.util import LOCALTZ
def _get_mappings(self):
    mappings = []
    if self.mapping_file:
        with open(self.mapping_file) as fileobj:
            method_map, options_map = parse_mapping(fileobj)
        for path in self.input_paths:
            mappings.append((path, method_map, options_map))
    elif getattr(self.distribution, 'message_extractors', None):
        message_extractors = self.distribution.message_extractors
        for path, mapping in message_extractors.items():
            if isinstance(mapping, str):
                method_map, options_map = parse_mapping(StringIO(mapping))
            else:
                method_map, options_map = ([], {})
                for pattern, method, options in mapping:
                    method_map.append((pattern, method))
                    options_map[pattern] = options or {}
            mappings.append((path, method_map, options_map))
    else:
        for path in self.input_paths:
            mappings.append((path, DEFAULT_MAPPING, {}))
    return mappings