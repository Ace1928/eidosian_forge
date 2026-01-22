import builtins
import codecs
import datetime
import select
import struct
import sys
import zlib
import subunit
import iso8601
def _write_number(self, value, packet):
    packet.extend(self._encode_number(value))