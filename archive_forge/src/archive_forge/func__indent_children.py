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
def _indent_children(elem, level):
    child_level = level + 1
    try:
        child_indentation = indentations[child_level]
    except IndexError:
        child_indentation = indentations[level] + space
        indentations.append(child_indentation)
    if not elem.text or not elem.text.strip():
        elem.text = child_indentation
    for child in elem:
        if len(child):
            _indent_children(child, child_level)
        if not child.tail or not child.tail.strip():
            child.tail = child_indentation
    if not child.tail.strip():
        child.tail = indentations[level]