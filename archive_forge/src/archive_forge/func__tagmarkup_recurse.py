from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def _tagmarkup_recurse(tm, attr):
    """Return (text list, attribute list) for tagmarkup passed.

    tm -- tagmarkup
    attr -- current attribute or None"""
    if isinstance(tm, list):
        rtl = []
        ral = []
        for element in tm:
            tl, al = _tagmarkup_recurse(element, attr)
            if ral:
                last_attr, last_run = ral[-1]
                top_attr, top_run = al[0]
                if last_attr == top_attr:
                    ral[-1] = (top_attr, last_run + top_run)
                    del al[0]
            rtl += tl
            ral += al
        return (rtl, ral)
    if isinstance(tm, tuple):
        if len(tm) != 2:
            raise TagMarkupException(f'Tuples must be in the form (attribute, tagmarkup): {tm!r}')
        attr, element = tm
        return _tagmarkup_recurse(element, attr)
    if not isinstance(tm, (str, bytes)):
        raise TagMarkupException(f'Invalid markup element: {tm!r}')
    return ([tm], [(attr, len(tm))])