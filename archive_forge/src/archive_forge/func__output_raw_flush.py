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
def _output_raw_flush(self, stream: 'StreamLiterals', data: Optional[str]=None) -> None:
    if data is None:
        output_raw = self._output_raw_streams[stream]
        try:
            data = output_raw._emulator.read()
        except Exception as e:
            logger.warning(f'problem reading from output_raw emulator: {e}')
    if data:
        self._send_output_line(stream, data)
        if self._output_raw_file:
            self._output_raw_file.write(data.encode('utf-8'))