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
def _config_artifact_callback(self, key: str, val: Union[str, Artifact, dict]) -> Artifact:
    if _is_artifact_version_weave_dict(val):
        assert isinstance(val, dict)
        public_api = self._public_api()
        artifact = Artifact._from_id(val['id'], public_api.client)
        return self.use_artifact(artifact, use_as=key)
    elif _is_artifact_string(val):
        assert isinstance(val, str)
        artifact_string, base_url, is_id = parse_artifact_string(val)
        overrides = {}
        if base_url is not None:
            overrides = {'base_url': base_url}
            public_api = public.Api(overrides)
        else:
            public_api = self._public_api()
        if is_id:
            artifact = Artifact._from_id(artifact_string, public_api._client)
        else:
            artifact = public_api.artifact(name=artifact_string)
        return self.use_artifact(artifact, use_as=key)
    elif _is_artifact_object(val):
        return self.use_artifact(val, use_as=key)
    else:
        raise ValueError(f'Cannot call _config_artifact_callback on type {type(val)}')