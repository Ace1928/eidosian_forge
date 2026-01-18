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
def tostringlist(element, encoding=None, method=None, *, xml_declaration=None, default_namespace=None, short_empty_elements=True):
    lst = []
    stream = _ListDataStream(lst)
    ElementTree(element).write(stream, encoding, xml_declaration=xml_declaration, default_namespace=default_namespace, method=method, short_empty_elements=short_empty_elements)
    return lst