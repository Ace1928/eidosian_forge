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
def add_composite_environment_options(parser: argparse.ArgumentParser, completer: CompositeActionCompletionFinder, controller_mode: ControllerMode, target_mode: TargetMode) -> list[t.Type[CompositeAction]]:
    """Add composite options for controlling the test environment."""
    composite_parser = t.cast(argparse.ArgumentParser, parser.add_argument_group(title='composite environment arguments (mutually exclusive with "environment arguments" above)'))
    composite_parser.add_argument('--host-path', help=argparse.SUPPRESS)
    action_types: list[t.Type[CompositeAction]] = []

    def register_action_type(action_type: t.Type[CompositeAction]) -> t.Type[CompositeAction]:
        """Register the provided composite action type and return it."""
        action_types.append(action_type)
        return action_type
    if controller_mode == ControllerMode.NO_DELEGATION:
        composite_parser.set_defaults(controller=None)
    else:
        register_completer(composite_parser.add_argument('--controller', metavar='OPT', action=register_action_type(DelegatedControllerAction if controller_mode == ControllerMode.DELEGATED else OriginControllerAction), help='configuration for the controller'), completer.completer)
    if target_mode == TargetMode.NO_TARGETS:
        composite_parser.set_defaults(targets=[])
    elif target_mode == TargetMode.SHELL:
        group = composite_parser.add_mutually_exclusive_group()
        register_completer(group.add_argument('--target-posix', metavar='OPT', action=register_action_type(PosixSshTargetAction), help='configuration for the target'), completer.completer)
        suppress = None if get_ci_provider().supports_core_ci_auth() else argparse.SUPPRESS
        register_completer(group.add_argument('--target-windows', metavar='OPT', action=WindowsSshTargetAction if suppress else register_action_type(WindowsSshTargetAction), help=suppress or 'configuration for the target'), completer.completer)
        register_completer(group.add_argument('--target-network', metavar='OPT', action=NetworkSshTargetAction if suppress else register_action_type(NetworkSshTargetAction), help=suppress or 'configuration for the target'), completer.completer)
    else:
        if target_mode.multiple_pythons:
            target_option = '--target-python'
            target_help = 'configuration for the target python interpreter(s)'
        elif target_mode == TargetMode.POSIX_INTEGRATION:
            target_option = '--target'
            target_help = 'configuration for the target'
        else:
            target_option = '--target'
            target_help = 'configuration for the target(s)'
        target_actions = {TargetMode.POSIX_INTEGRATION: PosixTargetAction, TargetMode.WINDOWS_INTEGRATION: WindowsTargetAction, TargetMode.NETWORK_INTEGRATION: NetworkTargetAction, TargetMode.SANITY: SanityPythonTargetAction, TargetMode.UNITS: UnitsPythonTargetAction}
        target_action = target_actions[target_mode]
        register_completer(composite_parser.add_argument(target_option, metavar='OPT', action=register_action_type(target_action), help=target_help), completer.completer)
    return action_types