from __future__ import unicode_literals
import math
import textwrap
import re
from cmakelang import common
class CommentItem(object):

    def __init__(self, kind):
        self.kind = kind
        self.indent = None
        self.lines = []

    def __repr__(self):
        return '{}'.format(self.kind.name)