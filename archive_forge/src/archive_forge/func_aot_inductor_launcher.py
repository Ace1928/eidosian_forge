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
def aot_inductor_launcher(so_path: str, device: str):
    if device == 'cuda':
        return f'\n            #include <torch/csrc/inductor/aoti_model_container_runner_cuda.h>\n\n            torch::inductor::AOTIModelContainerRunnerCuda runner("{so_path}");\n\n            std::vector<at::Tensor> run(std::vector<at::Tensor>& input_tensors) {{\n                return runner.run(input_tensors);\n            }}\n\n            std::vector<const char*> get_call_spec() {{\n                return runner.get_call_spec();\n            }}\n        '
    elif device == 'cpu':
        return f'\n            #include <torch/csrc/inductor/aoti_model_container_runner.h>\n\n            torch::inductor::AOTIModelContainerRunnerCpu runner("{so_path}");\n\n            std::vector<at::Tensor> run(std::vector<at::Tensor>& input_tensors) {{\n                return runner.run(input_tensors);\n            }}\n\n            std::vector<const char*> get_call_spec() {{\n                return runner.get_call_spec();\n            }}\n        '
    else:
        raise RuntimeError(f'Unsupported device: {device}')