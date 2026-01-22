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
def _history_update_leaf(self, kl: List[str], v: Any, history_dict: Dict[str, Any], update_history: Dict[str, Any]) -> None:
    hkey = '.'.join([k.replace('.', '\\.') for k in kl])
    m = self._metric_defines.get(hkey)
    if not m:
        m = self._history_define_metric(hkey)
        if not m:
            return
        mr = Record()
        mr.metric.CopyFrom(m)
        mr.control.local = True
        self._handle_defined_metric(mr)
    if m.options.step_sync and m.step_metric:
        if m.step_metric not in history_dict:
            copy_key = tuple([m.step_metric])
            step = self._metric_copy.get(copy_key)
            if step is not None:
                update_history[m.step_metric] = step