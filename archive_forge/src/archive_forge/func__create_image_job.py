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
def _create_image_job(self, input_types: Dict[str, Any], output_types: Dict[str, Any], installed_packages_list: List[str], docker_image_name: Optional[str]=None, args: Optional[List[str]]=None) -> Optional['Artifact']:
    docker_image_name = docker_image_name or os.getenv('WANDB_DOCKER')
    if not docker_image_name:
        return None
    name = wandb.util.make_artifact_name_safe(f'job-{docker_image_name}')
    s_args: Sequence[str] = args if args is not None else self._settings._args
    source_info: JobSourceDict = {'_version': 'v0', 'source_type': 'image', 'source': {'image': docker_image_name, 'args': s_args}, 'input_types': input_types, 'output_types': output_types, 'runtime': self._settings._python}
    job_artifact = self._construct_job_artifact(name, source_info, installed_packages_list)
    return job_artifact