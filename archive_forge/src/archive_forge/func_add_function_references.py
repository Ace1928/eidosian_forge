import gc
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import types
import weakref
import json
from tempfile import NamedTemporaryFile
import torch
from torch.cuda._memory_viz import _frames_fmt, _block_extra
import atexit
import logging
def add_function_references():
    add_attrs('__defaults__', '__closure__', '__globals__', '__code__', '__name__', '__module__', '__doc____qualname__', '__annotations__', '__kwdefaults__')