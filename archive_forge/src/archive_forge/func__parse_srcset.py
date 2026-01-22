import contextlib
import doctest
from io import StringIO
import itertools
import os
from os.path import relpath
from pathlib import Path
import re
import shutil
import sys
import textwrap
import traceback
from docutils.parsers.rst import directives, Directive
from docutils.parsers.rst.directives.images import Image
import jinja2  # Sphinx dependency.
from sphinx.errors import ExtensionError
import matplotlib
from matplotlib.backend_bases import FigureManagerBase
import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers, cbook
def _parse_srcset(entries):
    """
    Parse srcset for multiples...
    """
    srcset = {}
    for entry in entries:
        entry = entry.strip()
        if len(entry) >= 2:
            mult = entry[:-1]
            srcset[float(mult)] = entry
        else:
            raise ExtensionError(f'srcset argument {entry!r} is invalid.')
    return srcset