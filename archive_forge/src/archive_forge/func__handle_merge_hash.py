import importlib.util
import logging
import os
import textwrap
from functools import partial
from optparse import SUPPRESS_HELP, Option, OptionGroup, OptionParser, Values
from textwrap import dedent
from typing import Any, Callable, Dict, Optional, Tuple
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.cli.parser import ConfigOptionParser
from pip._internal.exceptions import CommandError
from pip._internal.locations import USER_CACHE_DIR, get_src_prefix
from pip._internal.models.format_control import FormatControl
from pip._internal.models.index import PyPI
from pip._internal.models.target_python import TargetPython
from pip._internal.utils.hashes import STRONG_HASHES
from pip._internal.utils.misc import strtobool
def _handle_merge_hash(option: Option, opt_str: str, value: str, parser: OptionParser) -> None:
    """Given a value spelled "algo:digest", append the digest to a list
    pointed to in a dict by the algo name."""
    if not parser.values.hashes:
        parser.values.hashes = {}
    try:
        algo, digest = value.split(':', 1)
    except ValueError:
        parser.error(f'Arguments to {opt_str} must be a hash name followed by a value, like --hash=sha256:abcde...')
    if algo not in STRONG_HASHES:
        parser.error('Allowed hash algorithms for {} are {}.'.format(opt_str, ', '.join(STRONG_HASHES)))
    parser.values.hashes.setdefault(algo, []).append(digest)