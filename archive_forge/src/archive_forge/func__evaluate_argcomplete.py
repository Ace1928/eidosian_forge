from __future__ import annotations
import argparse
import errno
import json
import os
import site
import sys
import sysconfig
from pathlib import Path
from shutil import which
from subprocess import Popen
from typing import Any
from . import paths
from .version import __version__
def _evaluate_argcomplete(parser: JupyterParser) -> list[str]:
    """If argcomplete is enabled, trigger autocomplete or return current words

    If the first word looks like a subcommand, return the current command
    that is attempting to be completed so that the subcommand can evaluate it;
    otherwise auto-complete using the main parser.
    """
    try:
        from traitlets.config.argcomplete_config import get_argcomplete_cwords, increment_argcomplete_index
        cwords = get_argcomplete_cwords()
        if cwords and len(cwords) > 1 and (not cwords[1].startswith('-')):
            increment_argcomplete_index()
            return cwords
        parser.argcomplete()
    except ImportError:
        parser.argcomplete()
    msg = 'Control flow should not reach end of autocomplete()'
    raise AssertionError(msg)