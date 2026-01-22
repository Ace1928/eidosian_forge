from __future__ import absolute_import
import re
import operator
import sys
def _get_first_or_none(it):
    try:
        try:
            _next = it.next
        except AttributeError:
            return next(it)
        else:
            return _next()
    except StopIteration:
        return None