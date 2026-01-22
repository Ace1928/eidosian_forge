import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
def _read_utilization(self):
    with self.lock:
        if psutil is not None:
            self.values['cpu_util_percent'].append(float(psutil.cpu_percent(interval=None)))
            self.values['ram_util_percent'].append(float(getattr(psutil.virtual_memory(), 'percent')))
        if self.GPUtil is not None:
            gpu_list = []
            try:
                gpu_list = self.GPUtil.getGPUs()
            except Exception:
                logger.debug('GPUtil failed to retrieve GPUs.')
            for gpu in gpu_list:
                self.values['gpu_util_percent' + str(gpu.id)].append(float(gpu.load))
                self.values['vram_util_percent' + str(gpu.id)].append(float(gpu.memoryUtil))