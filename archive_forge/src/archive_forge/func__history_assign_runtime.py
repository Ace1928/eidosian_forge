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
def _history_assign_runtime(self, history: HistoryRecord, history_dict: Dict[str, Any]) -> None:
    if '_timestamp' not in history_dict:
        return
    if self._run_start_time is None:
        self._run_start_time = history_dict['_timestamp']
    history_dict['_runtime'] = history_dict['_timestamp'] - self._run_start_time
    item = history.item.add()
    item.key = '_runtime'
    item.value_json = json.dumps(history_dict[item.key])