from __future__ import annotations
import argparse
import enum
import functools
import typing as t
from ..constants import (
from ..util import (
from ..completion import (
from ..cli.argparsing import (
from ..cli.argparsing.actions import (
from ..cli.actions import (
from ..cli.compat import (
from ..config import (
from .completers import (
from .converters import (
from .epilog import (
from ..ci import (
def add_environment_venv(exclusive_parser: argparse.ArgumentParser, environments_parser: argparse.ArgumentParser) -> None:
    """Add environment arguments for running in ansible-test managed virtual environments."""
    exclusive_parser.add_argument('--venv', action='store_true', help='run from a virtual environment')
    environments_parser.add_argument('--venv-system-site-packages', action='store_true', help='enable system site packages')