import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
class DotEdge(object):
    """Class representing a DOT edge"""

    def __init__(self, src, dst, directed=False, src_port='', dst_port='', **kwds):
        self.src = src
        self.dst = dst
        self.src_port = src_port
        self.dst_port = dst_port
        self.attr = {}
        if directed:
            self.conn = '->'
        else:
            self.conn = '--'
        self.attr.update(kwds)

    def __str__(self):
        attrstr = ','.join(['%s=%s' % (quote_if_necessary(key), quote_if_necessary(val)) for key, val in self.attr.items()])
        if attrstr:
            attrstr = '[%s]' % attrstr
        return '%s%s %s %s%s %s;\n' % (quote_if_necessary(self.src.name), self.src_port, self.conn, quote_if_necessary(self.dst.name), self.dst_port, attrstr)

    def get_source(self):
        return self.src.name

    def get_destination(self):
        return self.dst.name

    def __getattr__(self, name):
        try:
            return self.attr[name]
        except KeyError:
            raise AttributeError