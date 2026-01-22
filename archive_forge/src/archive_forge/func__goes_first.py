from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
def _goes_first(self, is_main: bool):
    if not is_main:
        self.wait_for_everyone()
    yield
    if is_main:
        self.wait_for_everyone()