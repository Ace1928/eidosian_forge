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
def _read_refs(self, n):
    return self._read_ints(n, self._ref_size)