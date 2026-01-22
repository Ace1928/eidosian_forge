import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
class DotSubGraph(DotGraph):
    """Class representing a DOT subgraph"""

    def __init__(self, name='subgG', strict=True, directed=False, **kwds):
        DotGraph.__init__(self, name, strict, directed, **kwds)