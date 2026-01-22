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
@staticmethod
def _event_listener(parent_pipe, signal_pipe, rank: int):
    logger.info('Starting event listener thread for rank %s', rank)
    while True:
        ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])
        if parent_pipe in ready_pipes:
            if parent_pipe.closed:
                logger.info('Pipe closed for process %s, stopping event listener thread', rank)
                return
            event = parent_pipe.recv()
            logger.info('Received event %s on process %s', event, rank)
            if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                with tempfile.NamedTemporaryFile(mode='r+') as tmp_file:
                    faulthandler.dump_traceback(tmp_file)
                    tmp_file.flush()
                    tmp_file.seek(0)
                    parent_pipe.send(tmp_file.read())
                    logger.info('Process %s sent traceback', rank)
        if signal_pipe in ready_pipes:
            return