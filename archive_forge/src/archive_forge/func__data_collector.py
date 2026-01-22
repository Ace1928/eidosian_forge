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
def _data_collector(self, level, prefix, visibility=None, docMode=False):
    if visibility is not None and visibility < self._visibility:
        return
    if prefix:
        yield (level, prefix, None, self)
        if level is not None:
            level += 1
    for cfg in self._data.values():
        yield from cfg._data_collector(level, cfg._name + ': ', visibility, docMode)