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
def verify_ddp_error_logged(model_DDP, err_substr):
    ddp_logging_data = model_DDP._get_ddp_logging_data()
    assert 'iteration' in ddp_logging_data
    assert 'has_error' in ddp_logging_data
    assert 'error' in ddp_logging_data
    logging_err = ddp_logging_data['error']
    actual = err_substr if err_substr.find('\nException raised from ') == -1 else err_substr.split('\nException raised from ')[0]
    assert actual in logging_err, f'Did not find expected {actual} in ddp logging data error: {logging_err}'