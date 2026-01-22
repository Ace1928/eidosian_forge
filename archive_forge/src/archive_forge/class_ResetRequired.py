import os
import textwrap
from enum import auto, Enum
from traceback import extract_stack, format_exc, format_list, StackSummary
from typing import cast, NoReturn, Optional
import torch._guards
from . import config
from .config import is_fbcode
from .utils import counters
import logging
class ResetRequired(TorchDynamoException):

    def __init__(self):
        super().__init__(textwrap.dedent('\n                Must call `torch._dynamo.reset()` before changing backends.  Detected two calls to\n                `torch.compile()` with a different backend compiler arguments.\n                '))