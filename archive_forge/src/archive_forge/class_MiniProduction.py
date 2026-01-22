import re
import types
import sys
import os.path
import inspect
import base64
import warnings
class MiniProduction(object):

    def __init__(self, str, name, len, func, file, line):
        self.name = name
        self.len = len
        self.func = func
        self.callable = None
        self.file = file
        self.line = line
        self.str = str

    def __str__(self):
        return self.str

    def __repr__(self):
        return 'MiniProduction(%s)' % self.str

    def bind(self, pdict):
        if self.func:
            self.callable = pdict[self.func]