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
@_run_decorator._noop_on_finish()
@_run_decorator._attach
def upsert_artifact(self, artifact_or_path: Union[Artifact, str], name: Optional[str]=None, type: Optional[str]=None, aliases: Optional[List[str]]=None, distributed_id: Optional[str]=None) -> Artifact:
    """Declare (or append to) a non-finalized artifact as output of a run.

        Note that you must call run.finish_artifact() to finalize the artifact.
        This is useful when distributed jobs need to all contribute to the same artifact.

        Arguments:
            artifact_or_path: (str or Artifact) A path to the contents of this artifact,
                can be in the following forms:
                    - `/local/directory`
                    - `/local/directory/file.txt`
                    - `s3://bucket/path`
                You can also pass an Artifact object created by calling
                `wandb.Artifact`.
            name: (str, optional) An artifact name. May be prefixed with entity/project.
                Valid names can be in the following forms:
                    - name:version
                    - name:alias
                    - digest
                This will default to the basename of the path prepended with the current
                run id  if not specified.
            type: (str) The type of artifact to log, examples include `dataset`, `model`
            aliases: (list, optional) Aliases to apply to this artifact,
                defaults to `["latest"]`
            distributed_id: (string, optional) Unique string that all distributed jobs share. If None,
                defaults to the run's group name.

        Returns:
            An `Artifact` object.
        """
    if self._get_group() == '' and distributed_id is None:
        raise TypeError('Cannot upsert artifact unless run is in a group or distributed_id is provided')
    if distributed_id is None:
        distributed_id = self._get_group()
    return self._log_artifact(artifact_or_path, name=name, type=type, aliases=aliases, distributed_id=distributed_id, finalize=False)