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
def add_frame_references():
    f_locals = obj.f_locals
    add_attrs('f_back', 'f_code', 'f_builtins', 'f_globals', 'f_trace', 'f_locals')
    if type(f_locals) is dict:
        for name, local in obj.f_locals.items():
            add_reference(f'local {name}', local)