from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
def _emulate_setname(subclass):
    for key, value in subclass.__dict__.items():
        if key.startswith('_'):
            continue
        if hasattr(value, '__set_name__'):
            getattr(value, '__set_name__')(subclass, key)