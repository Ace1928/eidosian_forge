import io
import json
import platform
import re
import sys
import tokenize
import traceback
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from enum import Enum
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import (
import click
from click.core import ParameterSource
from mypy_extensions import mypyc_attr
from pathspec import PathSpec
from pathspec.patterns.gitwildmatch import GitWildMatchPatternError
from _black_version import version as __version__
from black.cache import Cache
from black.comments import normalize_fmt_off
from black.const import (
from black.files import (
from black.handle_ipynb_magics import (
from black.linegen import LN, LineGenerator, transform_line
from black.lines import EmptyLineTracker, LinesBlock
from black.mode import FUTURE_FLAG_TO_FEATURE, VERSION_TO_FEATURES, Feature
from black.mode import Mode as Mode  # re-exported
from black.mode import Preview, TargetVersion, supports_feature
from black.nodes import (
from black.output import color_diff, diff, dump_to_file, err, ipynb_diff, out
from black.parsing import (  # noqa F401
from black.ranges import (
from black.report import Changed, NothingChanged, Report
from black.trans import iter_fexpr_spans
from blib2to3.pgen2 import token
from blib2to3.pytree import Leaf, Node
def validate_cell(src: str, mode: Mode) -> None:
    """Check that cell does not already contain TransformerManager transformations,
    or non-Python cell magics, which might cause tokenizer_rt to break because of
    indentations.

    If a cell contains ``!ls``, then it'll be transformed to
    ``get_ipython().system('ls')``. However, if the cell originally contained
    ``get_ipython().system('ls')``, then it would get transformed in the same way:

        >>> TransformerManager().transform_cell("get_ipython().system('ls')")
        "get_ipython().system('ls')
"
        >>> TransformerManager().transform_cell("!ls")
        "get_ipython().system('ls')
"

    Due to the impossibility of safely roundtripping in such situations, cells
    containing transformed magics will be ignored.
    """
    if any((transformed_magic in src for transformed_magic in TRANSFORMED_MAGICS)):
        raise NothingChanged
    if src[:2] == '%%' and src.split()[0][2:] not in PYTHON_CELL_MAGICS | mode.python_cell_magics:
        raise NothingChanged