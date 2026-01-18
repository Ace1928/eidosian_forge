from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_indicator(self, indicator, need_whitespace, whitespace=False, indention=False):
    if self.whitespace or not need_whitespace:
        data = indicator
    else:
        data = u' ' + indicator
    self.whitespace = whitespace
    self.indention = self.indention and indention
    self.column += len(data)
    self.open_ended = False
    if bool(self.encoding):
        data = data.encode(self.encoding)
    self.stream.write(data)