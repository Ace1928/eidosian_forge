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
def _join_processes(self, fn) -> None:
    timeout = get_timeout(self.id())
    start_time = time.time()
    subprocess_error = False
    try:
        while True:
            for i, p in enumerate(self.processes):
                if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                    print(f'Process {i} terminated with exit code {p.exitcode}, terminating remaining processes.')
                    active_children = torch.multiprocessing.active_children()
                    for ac in active_children:
                        ac.terminate()
                    subprocess_error = True
                    break
            if subprocess_error:
                break
            if all((p.exitcode is not None for p in self.processes)):
                break
            elapsed = time.time() - start_time
            if elapsed > timeout:
                self._get_timedout_process_traceback()
                print(f'Timing out after {timeout} seconds and killing subprocesses.')
                for p in self.processes:
                    p.terminate()
                break
            time.sleep(0.1)
        elapsed_time = time.time() - start_time
        if fn in self.skip_return_code_checks:
            self._check_no_test_errors(elapsed_time)
        else:
            self._check_return_codes(elapsed_time)
    finally:
        for pipe in self.pid_to_pipe.values():
            pipe.close()