import functools
import logging
import time
from typing import Any, Dict, List, Tuple
import torch
import torch.distributed as dist
from torch.distributed.logging_handlers import _log_handlers
def _exception_logger(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            msg_dict = _get_msg_dict(func.__name__, *args, **kwargs)
            msg_dict['error'] = f'{error}'
            _c10d_logger.debug(msg_dict)
            raise
    return wrapper