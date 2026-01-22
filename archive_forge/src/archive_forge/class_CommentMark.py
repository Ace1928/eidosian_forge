from __future__ import absolute_import
import warnings
import textwrap
from ruamel.yaml.compat import utf8
class CommentMark(object):
    __slots__ = ('column',)

    def __init__(self, column):
        self.column = column