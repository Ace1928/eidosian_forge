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
def add_environment_local(exclusive_parser: argparse.ArgumentParser) -> None:
    """Add environment arguments for running on the local (origin) host."""
    exclusive_parser.add_argument('--local', action='store_true', help='run from the local environment')