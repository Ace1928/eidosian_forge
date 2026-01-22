from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _convert_pycparser_error(self, e, csource):
    line = None
    msg = str(e)
    match = re.match('%s:(\\d+):' % (CDEF_SOURCE_STRING,), msg)
    if match:
        linenum = int(match.group(1), 10)
        csourcelines = csource.splitlines()
        if 1 <= linenum <= len(csourcelines):
            line = csourcelines[linenum - 1]
    return line