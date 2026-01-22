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
def _swap_artifact_name(self, artifact_name: str, use_as: Optional[str]) -> str:
    artifact_key_string = use_as or artifact_name
    replacement_artifact_info = self._launch_artifact_mapping.get(artifact_key_string)
    if replacement_artifact_info is not None:
        new_name = replacement_artifact_info.get('name')
        entity = replacement_artifact_info.get('entity')
        project = replacement_artifact_info.get('project')
        if new_name is None or entity is None or project is None:
            raise ValueError('Misconfigured artifact in launch config. Must include name, project and entity keys.')
        return f'{entity}/{project}/{new_name}'
    elif replacement_artifact_info is None and use_as is None:
        sequence_name = artifact_name.split(':')[0].split('/')[-1]
        unique_artifact_replacement_info = self._unique_launch_artifact_sequence_names.get(sequence_name)
        if unique_artifact_replacement_info is not None:
            new_name = unique_artifact_replacement_info.get('name')
            entity = unique_artifact_replacement_info.get('entity')
            project = unique_artifact_replacement_info.get('project')
            if new_name is None or entity is None or project is None:
                raise ValueError('Misconfigured artifact in launch config. Must include name, project and entity keys.')
            return f'{entity}/{project}/{new_name}'
    else:
        return artifact_name
    return artifact_name