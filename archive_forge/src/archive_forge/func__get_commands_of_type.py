from functools import partial
from typing import Literal
import click
from kombu.utils.json import dumps
from celery.bin.base import COMMA_SEPARATED_LIST, CeleryCommand, CeleryOption, handle_preload_options
from celery.exceptions import CeleryCommandException
from celery.platforms import EX_UNAVAILABLE
from celery.utils import text
from celery.worker.control import Panel
def _get_commands_of_type(type_: _RemoteControlType) -> dict:
    command_name_info_pairs = [(name, info) for name, info in Panel.meta.items() if info.type == type_ and info.visible]
    return dict(sorted(command_name_info_pairs))