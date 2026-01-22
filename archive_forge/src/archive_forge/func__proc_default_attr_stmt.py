import re
import itertools
import os
import logging
import string
import pyparsing
from pyparsing import __version__ as pyparsing_version
from pyparsing import (Literal, CaselessLiteral, Word, OneOrMore, Forward, Group, Optional, Combine, restOfLine,
from collections import OrderedDict
def _proc_default_attr_stmt(self, toks):
    """Return (ADD_DEFAULT_NODE_ATTR,options"""
    if len(toks) == 1:
        gtype = toks
        attr = {}
    else:
        gtype, attr = toks
    if gtype == 'node':
        return (SET_DEF_NODE_ATTR, attr)
    elif gtype == 'edge':
        return (SET_DEF_EDGE_ATTR, attr)
    elif gtype == 'graph':
        return (SET_DEF_GRAPH_ATTR, attr)
    else:
        return ('unknown', toks)