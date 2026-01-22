import faulthandler
import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial, reduce, wraps
from io import StringIO
from typing import Dict, NamedTuple, Optional, Union
from unittest.mock import patch
import torch
import torch._dynamo.test_case
import torch.cuda.nccl
import torch.distributed as c10d
import torch.nn as nn
from torch.testing._internal.common_utils import (
from torch.testing._internal.distributed.multi_threaded_pg import (
def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
    """
        The reason why we have this util function instead of
        self.assertEqual is all threads are sharing one CPU RNG
        so the assertion result is only reliable on rank 0
        """
    if self.rank == rank:
        self.assertEqual(x, y, msg)