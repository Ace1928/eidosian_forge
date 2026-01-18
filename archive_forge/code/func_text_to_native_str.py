import types
import sys
import numbers
import functools
import copy
import inspect
def text_to_native_str(t, encoding='ascii'):
    """
        Use this to create a Py2 native string when "from __future__ import
        unicode_literals" is in effect.
        """
    return unicode(t).encode(encoding)