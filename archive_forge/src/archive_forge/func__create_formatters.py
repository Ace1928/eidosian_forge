import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
def _create_formatters(cp):
    """Create and return formatters"""
    flist = cp['formatters']['keys']
    if not len(flist):
        return {}
    flist = flist.split(',')
    flist = _strip_spaces(flist)
    formatters = {}
    for form in flist:
        sectname = 'formatter_%s' % form
        fs = cp.get(sectname, 'format', raw=True, fallback=None)
        dfs = cp.get(sectname, 'datefmt', raw=True, fallback=None)
        stl = cp.get(sectname, 'style', raw=True, fallback='%')
        c = logging.Formatter
        class_name = cp[sectname].get('class')
        if class_name:
            c = _resolve(class_name)
        f = c(fs, dfs, stl)
        formatters[form] = f
    return formatters