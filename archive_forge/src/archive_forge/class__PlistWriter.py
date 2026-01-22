import binascii
import codecs
import datetime
import enum
from io import BytesIO
import itertools
import os
import re
import struct
from xml.parsers.expat import ParserCreate
class _PlistWriter(_DumbXMLWriter):

    def __init__(self, file, indent_level=0, indent=b'\t', writeHeader=1, sort_keys=True, skipkeys=False):
        if writeHeader:
            file.write(PLISTHEADER)
        _DumbXMLWriter.__init__(self, file, indent_level, indent)
        self._sort_keys = sort_keys
        self._skipkeys = skipkeys

    def write(self, value):
        self.writeln('<plist version="1.0">')
        self.write_value(value)
        self.writeln('</plist>')

    def write_value(self, value):
        if isinstance(value, str):
            self.simple_element('string', value)
        elif value is True:
            self.simple_element('true')
        elif value is False:
            self.simple_element('false')
        elif isinstance(value, int):
            if -1 << 63 <= value < 1 << 64:
                self.simple_element('integer', '%d' % value)
            else:
                raise OverflowError(value)
        elif isinstance(value, float):
            self.simple_element('real', repr(value))
        elif isinstance(value, dict):
            self.write_dict(value)
        elif isinstance(value, (bytes, bytearray)):
            self.write_bytes(value)
        elif isinstance(value, datetime.datetime):
            self.simple_element('date', _date_to_string(value))
        elif isinstance(value, (tuple, list)):
            self.write_array(value)
        else:
            raise TypeError('unsupported type: %s' % type(value))

    def write_bytes(self, data):
        self.begin_element('data')
        self._indent_level -= 1
        maxlinelength = max(16, 76 - len(self.indent.replace(b'\t', b' ' * 8) * self._indent_level))
        for line in _encode_base64(data, maxlinelength).split(b'\n'):
            if line:
                self.writeln(line)
        self._indent_level += 1
        self.end_element('data')

    def write_dict(self, d):
        if d:
            self.begin_element('dict')
            if self._sort_keys:
                items = sorted(d.items())
            else:
                items = d.items()
            for key, value in items:
                if not isinstance(key, str):
                    if self._skipkeys:
                        continue
                    raise TypeError('keys must be strings')
                self.simple_element('key', key)
                self.write_value(value)
            self.end_element('dict')
        else:
            self.simple_element('dict')

    def write_array(self, array):
        if array:
            self.begin_element('array')
            for value in array:
                self.write_value(value)
            self.end_element('array')
        else:
            self.simple_element('array')