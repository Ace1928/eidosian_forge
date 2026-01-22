import torch._dynamo.test_case
import unittest.mock
import os
import contextlib
import torch._logging
import torch._logging._internal
from torch._dynamo.utils import LazyString
import logging
def append_setting(name, level):
    if isinstance(name, str) and isinstance(level, int) and (level in INT_TO_VERBOSITY):
        settings.append(INT_TO_VERBOSITY[level] + name)
        return
    else:
        raise ValueError('Invalid value for setting')