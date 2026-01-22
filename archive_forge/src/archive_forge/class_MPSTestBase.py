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
class MPSTestBase(DeviceTypeTestBase):
    device_type = 'mps'
    primary_device: ClassVar[str]

    @classmethod
    def get_primary_device(cls):
        return cls.primary_device

    @classmethod
    def get_all_devices(cls):
        prim_device = cls.get_primary_device()
        return [prim_device]

    @classmethod
    def setUpClass(cls):
        cls.primary_device = 'mps:0'

    def _should_stop_test_suite(self):
        return False