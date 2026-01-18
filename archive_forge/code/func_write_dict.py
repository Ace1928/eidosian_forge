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