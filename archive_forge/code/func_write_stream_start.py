from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_stream_start(self):
    if self.encoding and self.encoding.startswith('utf-16'):
        self.stream.write(u'\ufeff'.encode(self.encoding))