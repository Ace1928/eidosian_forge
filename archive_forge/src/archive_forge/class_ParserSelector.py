import os
import re
import sys
import traceback
import ast
import importlib
from re import sub, findall
from types import CodeType
from functools import partial
from collections import OrderedDict, defaultdict
import kivy.lang.builder  # imported as absolute to avoid circular import
from kivy.logger import Logger
from kivy.cache import Cache
from kivy import require
from kivy.resources import resource_find
from kivy.utils import rgba
import kivy.metrics as Metrics
class ParserSelector(object):

    def __init__(self, key):
        self.key = key.lower()

    def match(self, widget):
        raise NotImplementedError

    def __repr__(self):
        return '<%s key=%s>' % (self.__class__.__name__, self.key)