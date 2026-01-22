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
def add_environment_docker(exclusive_parser: argparse.ArgumentParser, environments_parser: argparse.ArgumentParser, target_mode: TargetMode) -> None:
    """Add environment arguments for running in docker containers."""
    if target_mode in (TargetMode.POSIX_INTEGRATION, TargetMode.SHELL):
        docker_images = sorted(filter_completion(docker_completion()))
    else:
        docker_images = sorted(filter_completion(docker_completion(), controller_only=True))
    register_completer(exclusive_parser.add_argument('--docker', metavar='IMAGE', nargs='?', const='default', help='run from a docker container'), functools.partial(complete_choices, docker_images))
    environments_parser.add_argument('--docker-privileged', action='store_true', help='run docker container in privileged mode')
    environments_parser.add_argument('--docker-seccomp', metavar='SC', choices=SECCOMP_CHOICES, help='set seccomp confinement for the test container: %(choices)s')
    environments_parser.add_argument('--docker-memory', metavar='INT', type=int, help='memory limit for docker in bytes')