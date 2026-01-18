from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_double_quoted(self, text, split=True):
    if self.root_context:
        if self.requested_indent is not None:
            self.write_line_break()
            if self.requested_indent != 0:
                self.write_indent()
    self.write_indicator(u'"', True)
    start = end = 0
    while end <= len(text):
        ch = None
        if end < len(text):
            ch = text[end]
        if ch is None or ch in u'"\\\x85\u2028\u2029\ufeff' or (not (u' ' <= ch <= u'~' or (self.allow_unicode and (u'\xa0' <= ch <= u'\ud7ff' or u'\ue000' <= ch <= u'�')))):
            if start < end:
                data = text[start:end]
                self.column += len(data)
                if bool(self.encoding):
                    data = data.encode(self.encoding)
                self.stream.write(data)
                start = end
            if ch is not None:
                if ch in self.ESCAPE_REPLACEMENTS:
                    data = u'\\' + self.ESCAPE_REPLACEMENTS[ch]
                elif ch <= u'ÿ':
                    data = u'\\x%02X' % ord(ch)
                elif ch <= u'\uffff':
                    data = u'\\u%04X' % ord(ch)
                else:
                    data = u'\\U%08X' % ord(ch)
                self.column += len(data)
                if bool(self.encoding):
                    data = data.encode(self.encoding)
                self.stream.write(data)
                start = end + 1
        if 0 < end < len(text) - 1 and (ch == u' ' or start >= end) and (self.column + (end - start) > self.best_width) and split:
            data = text[start:end] + u'\\'
            if start < end:
                start = end
            self.column += len(data)
            if bool(self.encoding):
                data = data.encode(self.encoding)
            self.stream.write(data)
            self.write_indent()
            self.whitespace = False
            self.indention = False
            if text[start] == u' ':
                data = u'\\'
                self.column += len(data)
                if bool(self.encoding):
                    data = data.encode(self.encoding)
                self.stream.write(data)
        end += 1
    self.write_indicator(u'"', False)