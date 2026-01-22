import json
import os
import signal
import socket
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from torch.distributed.elastic.utils.logging import get_logger
from .error_handler import ErrorHandler  # noqa: F401
from .handlers import get_error_handler  # noqa: F401
def _get_error_data(self, error_file_data: Dict[str, Any]) -> Tuple[str, int]:
    message = error_file_data['message']
    if isinstance(message, str):
        timestamp = int(error_file_data.get('timestamp', 0))
    else:
        timestamp = int(message['extraInfo']['timestamp'])
    return (message, timestamp)