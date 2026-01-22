import copyreg
import functools
import sys
import traceback
import warnings
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, DefaultDict, List, Optional
import torch
class KeyErrorMessage(str):
    """str subclass that returns itself in repr"""

    def __repr__(self):
        return self