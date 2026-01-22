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
def _on_finish(self) -> None:
    trigger.call('on_finished')
    if self._run_status_checker is not None:
        self._run_status_checker.stop()
    self._console_stop()
    assert self._backend and self._backend.interface
    exit_handle = self._backend.interface.deliver_exit(self._exit_code)
    exit_handle.add_probe(on_probe=self._on_probe_exit)
    _ = exit_handle.wait(timeout=-1, on_progress=self._on_progress_exit)
    poll_exit_handle = self._backend.interface.deliver_poll_exit()
    result = poll_exit_handle.wait(timeout=-1)
    assert result
    self._footer_file_pusher_status_info(result.response.poll_exit_response, printer=self._printer)
    self._poll_exit_response = result.response.poll_exit_response
    internal_messages_handle = self._backend.interface.deliver_internal_messages()
    result = internal_messages_handle.wait(timeout=-1)
    assert result
    self._internal_messages_response = result.response.internal_messages_response
    server_info_handle = self._backend.interface.deliver_request_server_info()
    final_summary_handle = self._backend.interface.deliver_get_summary()
    sampled_history_handle = self._backend.interface.deliver_request_sampled_history()
    job_info_handle = self._backend.interface.deliver_request_job_info()
    result = server_info_handle.wait(timeout=-1)
    assert result
    self._server_info_response = result.response.server_info_response
    result = sampled_history_handle.wait(timeout=-1)
    assert result
    self._sampled_history = result.response.sampled_history_response
    result = final_summary_handle.wait(timeout=-1)
    assert result
    self._final_summary = result.response.get_summary_response
    result = job_info_handle.wait(timeout=-1)
    assert result
    self._job_info = result.response.job_info_response
    if self._backend:
        self._backend.cleanup()
    if self._run_status_checker:
        self._run_status_checker.join()
    self._unregister_telemetry_import_hooks(self._run_id)