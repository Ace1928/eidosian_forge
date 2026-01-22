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
def add_global_remote(parser: argparse.ArgumentParser, controller_mode: ControllerMode) -> None:
    """Add global options for remote instances."""
    if controller_mode != ControllerMode.DELEGATED:
        parser.set_defaults(remote_stage=None, remote_endpoint=None, remote_terminate=None)
        return
    suppress = None if get_ci_provider().supports_core_ci_auth() else argparse.SUPPRESS
    register_completer(parser.add_argument('--remote-stage', metavar='STAGE', default='prod', help=suppress or 'remote stage to use: prod, dev'), complete_remote_stage)
    parser.add_argument('--remote-endpoint', metavar='EP', help=suppress or 'remote provisioning endpoint to use')
    parser.add_argument('--remote-terminate', metavar='T', default=TerminateMode.NEVER, type=TerminateMode, action=EnumAction, help=suppress or 'terminate the remote instance: %(choices)s (default: %(default)s)')