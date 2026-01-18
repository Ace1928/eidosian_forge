import collections
import enum
import dataclasses
import itertools as it
import os
import pickle
import re
import shutil
import subprocess
import sys
import textwrap
from typing import (
import torch
from torch.utils.benchmark.utils import common, cpp_jit
from torch.utils.benchmark.utils._stubs import CallgrindModuleType
@staticmethod
def unwrap_all(globals: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v.value if isinstance(v, CopyIfCallgrind) else v for k, v in globals.items()}