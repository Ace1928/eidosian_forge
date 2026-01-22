from __future__ import annotations
import os
import socket
from functools import partial
from kombu.entity import Exchange, Queue
from .functional import memoize
from .text import simple_format
def _fmt_process_index(prefix: str='', default: str='0') -> str:
    from .log import current_process_index
    index = current_process_index()
    return f'{prefix}{index}' if index else default