import argparse
import functools
import itertools
import marshal
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List
@dataclass
class FrozenModule:
    module_name: str
    c_name: str
    size: int
    bytecode: bytes