from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
class TemplateObjectGetter(object):

    def __init__(self, template_obj):
        self.__template_obj = template_obj

    def __getattr__(self, attr):
        return getattr(self.__template_obj, attr, Empty)

    def __repr__(self):
        return '<%s around %r>' % (self.__class__.__name__, self.__template_obj)