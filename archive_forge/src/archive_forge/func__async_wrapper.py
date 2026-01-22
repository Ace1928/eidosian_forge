import enum
import os
import socket
import subprocess
import uuid
from platform import uname
from typing import List, Tuple, Union
from packaging.version import parse, Version
import psutil
import torch
import asyncio
from functools import partial
from typing import (
from collections import OrderedDict
from typing import Any, Hashable, Optional
from vllm.logger import init_logger
def _async_wrapper(*args, **kwargs) -> asyncio.Future:
    loop = asyncio.get_event_loop()
    p_func = partial(func, *args, **kwargs)
    return loop.run_in_executor(executor=None, func=p_func)