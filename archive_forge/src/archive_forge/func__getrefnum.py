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
def _getrefnum(self, value):
    if isinstance(value, _scalars):
        return self._objtable[type(value), value]
    else:
        return self._objidtable[id(value)]