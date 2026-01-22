from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def _aslist(obj):
    """
    Turn object into a list; lists and tuples are left as-is, None
    becomes [], and everything else turns into a one-element list.
    """
    if obj is None:
        return []
    elif isinstance(obj, (list, tuple)):
        return obj
    else:
        return [obj]