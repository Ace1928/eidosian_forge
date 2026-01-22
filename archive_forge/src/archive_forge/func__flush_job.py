import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime
from queue import Queue
from typing import (
import requests
import wandb
from wandb import util
from wandb.errors import CommError, UsageError
from wandb.errors.util import ProtobufErrorHandler
from wandb.filesync.dir_watcher import DirWatcher
from wandb.proto import wandb_internal_pb2
from wandb.sdk.artifacts.artifact_saver import ArtifactSaver
from wandb.sdk.interface import interface
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import (
from wandb.sdk.internal.file_pusher import FilePusher
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import (
from wandb.sdk.lib.mailbox import ContextCancelledError
from wandb.sdk.lib.proto_util import message_to_dict
def _flush_job(self) -> None:
    if self._job_builder.disable or self._settings._offline:
        return
    self._job_builder.set_config(self._consolidated_config.non_internal_config())
    summary_dict = self._cached_summary.copy()
    summary_dict.pop('_wandb', None)
    self._job_builder.set_summary(summary_dict)
    artifact = self._job_builder.build()
    if artifact is not None and self._run is not None:
        proto_artifact = self._interface._make_artifact(artifact)
        proto_artifact.run_id = self._run.run_id
        proto_artifact.project = self._run.project
        proto_artifact.entity = self._run.entity
        proto_artifact.aliases.append('latest')
        for alias in self._job_builder._aliases:
            proto_artifact.aliases.append(alias)
        proto_artifact.user_created = True
        proto_artifact.use_after_commit = True
        proto_artifact.finalize = True
        self._interface._publish_artifact(proto_artifact)