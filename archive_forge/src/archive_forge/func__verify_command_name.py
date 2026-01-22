from functools import partial
from typing import Literal
import click
from kombu.utils.json import dumps
from celery.bin.base import COMMA_SEPARATED_LIST, CeleryCommand, CeleryOption, handle_preload_options
from celery.exceptions import CeleryCommandException
from celery.platforms import EX_UNAVAILABLE
from celery.utils import text
from celery.worker.control import Panel
def _verify_command_name(type_: _RemoteControlType, command: str) -> None:
    choices = _get_commands_of_type(type_)
    if command not in choices:
        command_listing = ', '.join(choices)
        raise click.UsageError(message=f'Command {command} not recognized. Available {type_} commands: {command_listing}')