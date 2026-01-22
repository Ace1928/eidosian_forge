import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
def _get_devices_properties(device_ids):
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]