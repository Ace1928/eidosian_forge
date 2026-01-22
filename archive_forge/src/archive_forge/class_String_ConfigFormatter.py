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
class String_ConfigFormatter(ConfigFormatter):

    def __init__(self, block_start, block_end, item_start, item_body, item_end):
        self._block_start = _formatter_str_to_callback(block_start, self)
        self._block_end = _formatter_str_to_callback(block_end, self)
        self._item_start = _formatter_str_to_callback(item_start, self)
        self._item_end = _formatter_str_to_callback(item_end, self)
        self._item_body = _formatter_str_to_item_callback(item_body, self)