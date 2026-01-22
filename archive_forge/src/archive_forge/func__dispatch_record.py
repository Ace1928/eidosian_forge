import json
import logging
import math
import numbers
import time
from collections import defaultdict
from queue import Queue
from threading import Event
from typing import (
from wandb.proto.wandb_internal_pb2 import (
from ..interface.interface_queue import InterfaceQueue
from ..lib import handler_util, proto_util, tracelog, wburls
from . import context, sample, tb_watcher
from .settings_static import SettingsStatic
from .system.system_monitor import SystemMonitor
def _dispatch_record(self, record: Record, always_send: bool=False) -> None:
    if always_send:
        record.control.always_send = True
    tracelog.log_message_queue(record, self._writer_q)
    self._writer_q.put(record)