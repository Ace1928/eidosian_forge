from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
class TemplateObject(object):

    def __init__(self, name):
        self.__name = name
        self.get = TemplateObjectGetter(self)

    def __repr__(self):
        return '<%s %s>' % (self.__class__.__name__, self.__name)