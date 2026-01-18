from __future__ import annotations
import abc
import contextlib
import dataclasses
import difflib
import io
import logging
import sys
from typing import Any, Callable, Optional, Tuple
import torch
import torch.fx
from torch._subclasses import fake_tensor
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
from torch.onnx._internal import _beartype
from torch.onnx._internal.fx import diagnostics, onnxfunction_dispatcher
def to_onnx_domain_string(self) -> str:
    return '.'.join(filter(None, ('pkg', self.package_name, self.version, self.commit_hash)))