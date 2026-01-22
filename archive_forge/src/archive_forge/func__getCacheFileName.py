import sys, os, pickle
from hashlib import md5
from xml.sax.saxutils import quoteattr
from time import process_time as clock
from reportlab.lib.utils import asBytes, asNative as _asNative
from reportlab.lib.utils import rl_isdir, rl_isfile, rl_listdir, rl_getmtime
def _getCacheFileName(self):
    """Base this on the directories...same set of directories
        should give same cache"""
    fsEncoding = self._fsEncoding
    hash = md5(b''.join((asBytes(_, enc=fsEncoding) for _ in sorted(self._dirs)))).hexdigest()
    from reportlab.lib.utils import get_rl_tempfile
    fn = get_rl_tempfile('fonts_%s.dat' % hash)
    return fn