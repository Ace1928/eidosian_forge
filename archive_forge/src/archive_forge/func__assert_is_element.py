import sys
import re
import warnings
import io
import collections
import collections.abc
import contextlib
import weakref
from . import ElementPath
fromstring = XML
def _assert_is_element(self, e):
    if not isinstance(e, _Element_Py):
        raise TypeError('expected an Element, not %s' % type(e).__name__)