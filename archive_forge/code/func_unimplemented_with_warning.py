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
def unimplemented_with_warning(e: Exception, code, msg: str) -> NoReturn:
    graph_break_msg = format_error_msg_verbose(e, code)
    graph_breaks_log.debug('%s', graph_break_msg)
    log.warning(msg)
    raise unimplemented(msg) from e