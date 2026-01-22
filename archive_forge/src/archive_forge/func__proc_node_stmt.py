import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def _proc_node_stmt(self, toks):
    """Return (ADD_NODE, node_name, options)"""
    if len(toks) == 2:
        return tuple([ADD_NODE] + list(toks))
    else:
        return tuple([ADD_NODE] + list(toks) + [{}])