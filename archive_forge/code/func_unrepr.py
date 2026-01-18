import builtins
import configparser
import operator
import sys
from cherrypy._cpcompat import text_or_bytes
def unrepr(s):
    """Return a Python object compiled from a string."""
    if not s:
        return s
    b = _Builder()
    obj = b.astnode(s)
    return b.build(obj)