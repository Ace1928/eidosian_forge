from types import FunctionType
from copyreg import dispatch_table
from copyreg import _extension_registry, _inverted_registry, _extension_cache
from itertools import islice
from functools import partial
import sys
from sys import maxsize
from struct import pack, unpack
import re
import io
import codecs
import _compat_pickle
def write_large_bytes(self, header, payload):
    write = self.file_write
    if self.current_frame:
        self.commit_frame(force=True)
    write(header)
    write(payload)