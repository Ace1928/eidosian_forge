import argparse
import builtins
import enum
import importlib
import inspect
import io
import logging
import os
import pickle
import ply.lex
import re
import sys
import textwrap
import types
from operator import attrgetter
from pyomo.common.collections import Sequence, Mapping
from pyomo.common.deprecation import (
from pyomo.common.fileutils import import_file
from pyomo.common.formatting import wrap_reStructuredText
from pyomo.common.modeling import NOTSET
def _process_argparse_def(obj, _args, _kwds):
    _parser = parser
    _kwds = dict(_kwds)
    if 'group' in _kwds:
        _group = _kwds.pop('group')
        if isinstance(_group, tuple):
            for _idx, _grp in enumerate(_group):
                _issub, _parser = _get_subparser_or_group(_parser, _grp)
                if not _issub and _idx < len(_group) - 1:
                    raise RuntimeError("Could not find argparse subparser '%s' for Config item %s" % (_grp, obj.name(True)))
        else:
            _issub, _parser = _get_subparser_or_group(_parser, _group)
    if 'dest' not in _kwds:
        _kwds['dest'] = 'CONFIGBLOCK.' + obj.name(True)
        if 'metavar' not in _kwds and _kwds.get('action', '') not in _store_bool and (obj._domain is not None):
            _kwds['metavar'] = obj.domain_name().upper()
    _parser.add_argument(*_args, default=argparse.SUPPRESS, **_kwds)