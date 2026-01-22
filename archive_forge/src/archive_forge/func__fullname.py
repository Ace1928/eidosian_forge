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
@staticmethod
def _fullname(klass):
    """
        Get full name of class, including appropriate module qualifier.
        """
    module_name = klass.__module__
    module_qual = '' if module_name == 'builtins' else f'{module_name}.'
    return f'{module_qual}{klass.__name__}'