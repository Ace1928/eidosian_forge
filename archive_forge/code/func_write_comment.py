from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def write_comment(self, s):
    self.f.write(f'% {s}\n'.encode())