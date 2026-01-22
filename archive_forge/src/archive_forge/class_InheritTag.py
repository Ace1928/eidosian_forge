import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
class InheritTag(Tag):
    __keyword__ = 'inherit'

    def __init__(self, keyword, attributes, **kwargs):
        super().__init__(keyword, attributes, ('file',), (), ('file',), **kwargs)