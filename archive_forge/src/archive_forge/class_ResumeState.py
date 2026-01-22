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
class ResumeState:
    resumed: bool
    step: int
    history: int
    events: int
    output: int
    runtime: float
    wandb_runtime: Optional[int]
    summary: Optional[Dict[str, Any]]
    config: Optional[Dict[str, Any]]
    tags: Optional[List[str]]

    def __init__(self) -> None:
        self.resumed = False
        self.step = 0
        self.history = 0
        self.events = 0
        self.output = 0
        self.runtime = 0
        self.wandb_runtime = None
        self.summary = None
        self.config = None
        self.tags = None

    def __str__(self) -> str:
        obj = ','.join(map(lambda it: f'{it[0]}={it[1]}', vars(self).items()))
        return f'ResumeState({obj})'