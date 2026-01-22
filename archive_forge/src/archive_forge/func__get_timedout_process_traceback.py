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
def _get_timedout_process_traceback(self) -> None:
    pipes = []
    for i, process in enumerate(self.processes):
        if process.exitcode is None:
            pipe = self.pid_to_pipe[process.pid]
            try:
                pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                pipes.append((i, pipe))
            except ConnectionError as e:
                logger.error('Encountered error while trying to get traceback for process %s: %s', i, e)
    for rank, pipe in pipes:
        try:
            if pipe.poll(5):
                if pipe.closed:
                    logger.info('Pipe closed for process %s, cannot retrieve traceback', rank)
                    continue
                traceback = pipe.recv()
                logger.error('Process %s timed out with traceback: \n\n%s', rank, traceback)
            else:
                logger.error('Could not retrieve traceback for timed out process: %s', rank)
        except ConnectionError as e:
            logger.error('Encountered error while trying to get traceback for process %s: %s', rank, e)