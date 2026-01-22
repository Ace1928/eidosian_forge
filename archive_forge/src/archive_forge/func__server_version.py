import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def _server_version(engine):
    """Return a server_version_info tuple."""
    conn = engine.connect()
    version = getattr(engine.dialect, 'server_version_info', None)
    if version is None:
        version = ()
    conn.close()
    return version