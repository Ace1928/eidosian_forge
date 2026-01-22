import copy
import gc
import inspect
import runpy
import sys
import threading
from collections import namedtuple
from enum import Enum
from functools import wraps, partial
from typing import List, Any, ClassVar, Optional, Sequence, Tuple, Union, Dict, Set
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, \
from torch.testing._internal.common_cuda import _get_torch_cuda_version, \
from torch.testing._internal.common_dtype import get_all_dtypes
class deviceCountAtLeast:

    def __init__(self, num_required_devices):
        self.num_required_devices = num_required_devices

    def __call__(self, fn):
        assert not hasattr(fn, 'num_required_devices'), f'deviceCountAtLeast redefinition for {fn.__name__}'
        fn.num_required_devices = self.num_required_devices

        @wraps(fn)
        def multi_fn(slf, devices, *args, **kwargs):
            if len(devices) < self.num_required_devices:
                reason = f'fewer than {self.num_required_devices} devices detected'
                raise unittest.SkipTest(reason)
            return fn(slf, devices, *args, **kwargs)
        return multi_fn