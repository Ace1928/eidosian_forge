import keyword
import sys
import os
import types
import importlib
import pyparsing as pp
@classmethod
def trigger_url(cls):
    if cls.suffix is None:
        raise ValueError('%s.suffix is not set' % cls.__name__)
    return 'suffix:%s' % cls.suffix