from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def warn_unused(kwargs):
    unused = []
    for keyword, value in kwargs.items():
        if keyword.startswith('_'):
            continue
        if inspect.ismodule(value):
            continue
        unused.append(keyword)
    if unused:
        logger.warning('The following configuration options were ignored:\n  %s', '\n  '.join(unused))