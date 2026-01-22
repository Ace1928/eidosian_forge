import pickle
import sys
import os
import io
import subprocess
import json
from functools import lru_cache
from typing import Any
from itertools import groupby
import base64
import warnings
import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
def _block_extra(b):
    if 'frames' not in b:
        return _block_extra_legacy(b)
    return (b['frames'], b['requested_size'])