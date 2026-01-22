import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def _proc_attr_assignment(self, toks):
    return (SET_GRAPH_ATTR, dict(nsplit(toks, 2)))