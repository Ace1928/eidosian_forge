from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def write_obj(self, ref, *objs, **dict_obj):
    f = self.f
    if ref is None:
        ref = self.next_object_id(f.tell())
    else:
        self.xref_table[ref.object_id] = (f.tell(), ref.generation)
    f.write(bytes(IndirectObjectDef(*ref)))
    stream = dict_obj.pop('stream', None)
    if stream is not None:
        dict_obj['Length'] = len(stream)
    if dict_obj:
        f.write(pdf_repr(dict_obj))
    for obj in objs:
        f.write(pdf_repr(obj))
    if stream is not None:
        f.write(b'stream\n')
        f.write(stream)
        f.write(b'\nendstream\n')
    f.write(b'endobj\n')
    return ref