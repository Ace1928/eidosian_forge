from __future__ import absolute_import
from __future__ import print_function
import sys
from ruamel.yaml.error import YAMLError, YAMLStreamError
from ruamel.yaml.events import *  # NOQA
from ruamel.yaml.compat import utf8, text_type, PY2, nprint, dbg, DBG_EVENT, \
def write_single_quoted(self, text, split=True):
    if self.root_context:
        if self.requested_indent is not None:
            self.write_line_break()
            if self.requested_indent != 0:
                self.write_indent()
    self.write_indicator(u"'", True)
    spaces = False
    breaks = False
    start = end = 0
    while end <= len(text):
        ch = None
        if end < len(text):
            ch = text[end]
        if spaces:
            if ch is None or ch != u' ':
                if start + 1 == end and self.column > self.best_width and split and (start != 0) and (end != len(text)):
                    self.write_indent()
                else:
                    data = text[start:end]
                    self.column += len(data)
                    if bool(self.encoding):
                        data = data.encode(self.encoding)
                    self.stream.write(data)
                start = end
        elif breaks:
            if ch is None or ch not in u'\n\x85\u2028\u2029':
                if text[start] == u'\n':
                    self.write_line_break()
                for br in text[start:end]:
                    if br == u'\n':
                        self.write_line_break()
                    else:
                        self.write_line_break(br)
                self.write_indent()
                start = end
        elif ch is None or ch in u' \n\x85\u2028\u2029' or ch == u"'":
            if start < end:
                data = text[start:end]
                self.column += len(data)
                if bool(self.encoding):
                    data = data.encode(self.encoding)
                self.stream.write(data)
                start = end
        if ch == u"'":
            data = u"''"
            self.column += 2
            if bool(self.encoding):
                data = data.encode(self.encoding)
            self.stream.write(data)
            start = end + 1
        if ch is not None:
            spaces = ch == u' '
            breaks = ch in u'\n\x85\u2028\u2029'
        end += 1
    self.write_indicator(u"'", False)