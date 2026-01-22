import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
def add_callers(target, source):
    """Combine two caller lists in a single list."""
    new_callers = {}
    for func, caller in target.items():
        new_callers[func] = caller
    for func, caller in source.items():
        if func in new_callers:
            if isinstance(caller, tuple):
                new_callers[func] = tuple((i + j for i, j in zip(caller, new_callers[func])))
            else:
                new_callers[func] += caller
        else:
            new_callers[func] = caller
    return new_callers