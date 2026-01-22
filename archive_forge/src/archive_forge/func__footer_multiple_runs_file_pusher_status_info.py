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
def _footer_multiple_runs_file_pusher_status_info(poll_exit_responses: List[Optional[PollExitResponse]], *, printer: Union['PrinterTerm', 'PrinterJupyter']) -> None:
    if not all(poll_exit_responses):
        return
    megabyte = wandb.util.POW_2_BYTES[2][1]
    total_files: int = sum((sum([response.file_counts.wandb_count, response.file_counts.media_count, response.file_counts.artifact_count, response.file_counts.other_count]) for response in poll_exit_responses if response is not None and response.file_counts is not None))
    uploaded = sum((response.pusher_stats.uploaded_bytes for response in poll_exit_responses if response is not None and response.pusher_stats is not None))
    total = sum((response.pusher_stats.total_bytes for response in poll_exit_responses if response is not None and response.pusher_stats is not None))
    line = f'Processing {len(poll_exit_responses)} runs with {total_files} files ({uploaded / megabyte:.2f} MB/{total / megabyte:.2f} MB)\r'
    printer.progress_update(line)
    done = all([poll_exit_response.done for poll_exit_response in poll_exit_responses if poll_exit_response])
    if done:
        printer.progress_close()