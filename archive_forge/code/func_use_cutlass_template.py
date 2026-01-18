from __future__ import annotations
import collections
import contextlib
import enum
import functools
import getpass
import inspect
import itertools
import logging
import math
import operator
import os
import platform
import re
import shutil
import sys
import tempfile
import textwrap
import time
import unittest
from io import StringIO
from typing import (
from unittest import mock
import sympy
from typing_extensions import Concatenate, ParamSpec
import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch.autograd import DeviceType
from torch.autograd.profiler_util import EventList
from torch.utils._sympy.functions import CeilDiv, CleanDiv, FloorDiv, ModularIndexing
from . import config
def use_cutlass_template(layout):
    from .codegen.cuda.cutlass_utils import try_import_cutlass
    if torch.version.hip:
        return False
    layout_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    res = _use_template_for_cuda(layout, layout_dtypes) and _use_autotune_backend('CUTLASS')
    if res:
        if not try_import_cutlass():
            log.warning('Failed to import CUTLASS lib. Please check whether _inductor.config.cuda.cutlass_dir is set correctly. Skipping CUTLASS backend for now.')
            return False
    return res