from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _put_back_line_directives(csource, line_directives):

    def replace(m):
        s = m.group()
        if not s.startswith('#line@'):
            raise AssertionError('unexpected #line directive (should have been processed and removed')
        return line_directives[int(s[6:])]
    return _r_line_directive.sub(replace, csource)