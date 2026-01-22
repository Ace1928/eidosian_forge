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
class JupyterParser(argparse.ArgumentParser):
    """A Jupyter argument parser."""

    @property
    def epilog(self) -> str | None:
        """Add subcommands to epilog on request

        Avoids searching PATH for subcommands unless help output is requested.
        """
        return 'Available subcommands: %s' % ' '.join(list_subcommands())

    @epilog.setter
    def epilog(self, x: Any) -> None:
        """Ignore epilog set in Parser.__init__"""

    def argcomplete(self) -> None:
        """Trigger auto-completion, if enabled"""
        try:
            import argcomplete
            argcomplete.autocomplete(self)
        except ImportError:
            pass