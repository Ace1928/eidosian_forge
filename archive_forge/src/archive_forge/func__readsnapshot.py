import os
import sys
import py
import tempfile
def _readsnapshot(self, f):
    f.seek(0)
    res = f.read()
    enc = getattr(f, 'encoding', None)
    if enc:
        res = py.builtin._totext(res, enc, 'replace')
    f.truncate(0)
    f.seek(0)
    return res