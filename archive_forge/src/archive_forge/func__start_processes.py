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
def _start_processes(self, proc) -> None:
    self.processes = []
    for rank in range(int(self.world_size)):
        parent_conn, child_conn = torch.multiprocessing.Pipe()
        process = proc(target=self.__class__._run, name='process ' + str(rank), args=(rank, self._current_test_name(), self.file_name, child_conn))
        process.start()
        logger.info('Started process %s with pid %s', rank, process.pid)
        self.pid_to_pipe[process.pid] = parent_conn
        self.processes.append(process)