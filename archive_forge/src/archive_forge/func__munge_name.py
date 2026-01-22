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
def _munge_name(name, space_to_dash=True):
    if space_to_dash:
        name = re.sub('\\s', '-', name)
    name = re.sub('_', '-', name)
    return re.sub('[^a-zA-Z0-9-_]', '_', name)