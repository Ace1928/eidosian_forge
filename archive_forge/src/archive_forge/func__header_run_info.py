import _thread as thread
import atexit
import functools
import glob
import json
import logging
import numbers
import os
import re
import sys
import threading
import time
import traceback
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import IntEnum
from types import TracebackType
from typing import (
import requests
import wandb
import wandb.env
from wandb import errors, trigger
from wandb._globals import _datatypes_set_callback
from wandb.apis import internal, public
from wandb.apis.internal import Api
from wandb.apis.public import Api as PublicApi
from wandb.proto.wandb_internal_pb2 import (
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.internal import job_builder
from wandb.sdk.lib.import_hooks import (
from wandb.sdk.lib.paths import FilePathStr, LogicalPath, StrPath
from wandb.util import (
from wandb.viz import CustomChart, Visualize, custom_chart
from . import wandb_config, wandb_metric, wandb_summary
from .data_types._dtypes import TypeRegistry
from .interface.interface import GlobStr, InterfaceBase
from .interface.summary_record import SummaryRecord
from .lib import (
from .lib.exit_hooks import ExitHooks
from .lib.gitlib import GitRepo
from .lib.mailbox import MailboxError, MailboxHandle, MailboxProbe, MailboxProgress
from .lib.printer import get_printer
from .lib.proto_util import message_to_dict
from .lib.reporting import Reporter
from .lib.wburls import wburls
from .wandb_settings import Settings
from .wandb_setup import _WandbSetup
@staticmethod
def _header_run_info(*, settings: 'Settings', printer: Union['PrinterTerm', 'PrinterJupyter']) -> None:
    if settings._offline or settings.silent:
        return
    run_url = settings.run_url
    project_url = settings.project_url
    sweep_url = settings.sweep_url
    run_state_str = 'Resuming run' if settings.resumed else 'Syncing run'
    run_name = settings.run_name
    if not run_name:
        return
    if printer._html:
        if not wandb.jupyter.maybe_display():
            run_line = f'<strong>{printer.link(run_url, run_name)}</strong>'
            project_line, sweep_line = ('', '')
            if not wandb.jupyter.quiet():
                doc_html = printer.link(wburls.get('doc_run'), 'docs')
                project_html = printer.link(project_url, 'Weights & Biases')
                project_line = f'to {project_html} ({doc_html})'
                if sweep_url:
                    sweep_line = f'Sweep page: {printer.link(sweep_url, sweep_url)}'
            printer.display([f'{run_state_str} {run_line} {project_line}', sweep_line])
    else:
        printer.display(f'{run_state_str} {printer.name(run_name)}', off=not run_name)
    if not settings.quiet:
        printer.display(f'{printer.emoji('star')} View project at {printer.link(project_url)}')
        if sweep_url:
            printer.display(f'{printer.emoji('broom')} View sweep at {printer.link(sweep_url)}')
    printer.display(f'{printer.emoji('rocket')} View run at {printer.link(run_url)}')
    if Api().api.settings().get('anonymous') == 'true':
        printer.display('Do NOT share these links with anyone. They can be used to claim your runs.', level='warn', off=not run_name)