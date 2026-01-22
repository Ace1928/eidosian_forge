import sys
import os
import time
import marshal
import re
from enum import StrEnum, _simple_enum
from functools import cmp_to_key
from dataclasses import dataclass
from typing import Dict
@dataclass(unsafe_hash=True)
class StatsProfile:
    """Class for keeping track of an item in inventory."""
    total_tt: float
    func_profiles: Dict[str, FunctionProfile]