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
@property
def adjust_scheduler(self) -> bool:
    """Returns whether the scheduler should be adjusted"""
    return self.plugin_kwargs.get('adjust_scheduler', False)