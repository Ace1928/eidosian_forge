import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def WriteHmap(output_name, filelist):
    """Generates a header map based on |filelist|.

  Per Mark Mentovai:
    A header map is structured essentially as a hash table, keyed by names used
    in #includes, and providing pathnames to the actual files.

  The implementation below and the comment above comes from inspecting:
    http://www.opensource.apple.com/source/distcc/distcc-2503/distcc_dist/include_server/headermap.py?txt
  while also looking at the implementation in clang in:
    https://llvm.org/svn/llvm-project/cfe/trunk/lib/Lex/HeaderMap.cpp
  """
    magic = 1751998832
    version = 1
    _reserved = 0
    count = len(filelist)
    capacity = NextGreaterPowerOf2(count)
    strings_offset = 24 + 12 * capacity
    max_value_length = max((len(value) for value in filelist.values()))
    out = open(output_name, 'wb')
    out.write(struct.pack('<LHHLLLL', magic, version, _reserved, strings_offset, count, capacity, max_value_length))
    buckets = [None] * capacity
    for file, path in filelist.items():
        key = 0
        for c in file:
            key += ord(c.lower()) * 13
        while buckets[key & capacity - 1] is not None:
            key = key + 1
        buckets[key & capacity - 1] = (file, path)
    next_offset = 1
    for bucket in buckets:
        if bucket is None:
            out.write(struct.pack('<LLL', 0, 0, 0))
        else:
            file, path = bucket
            key_offset = next_offset
            prefix_offset = key_offset + len(file) + 1
            suffix_offset = prefix_offset + len(os.path.dirname(path) + os.sep) + 1
            next_offset = suffix_offset + len(os.path.basename(path)) + 1
            out.write(struct.pack('<LLL', key_offset, prefix_offset, suffix_offset))
    out.write(struct.pack('<x'))
    for bucket in buckets:
        if bucket is not None:
            file, path = bucket
            out.write(struct.pack('<%ds' % len(file), file))
            out.write(struct.pack('<s', '\x00'))
            base = os.path.dirname(path) + os.sep
            out.write(struct.pack('<%ds' % len(base), base))
            out.write(struct.pack('<s', '\x00'))
            path = os.path.basename(path)
            out.write(struct.pack('<%ds' % len(path), path))
            out.write(struct.pack('<s', '\x00'))