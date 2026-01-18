import collections
import contextlib
import dataclasses
import os
import shutil
import tempfile
import textwrap
import time
from typing import cast, Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Tuple
import uuid
import torch
def unit_to_english(u: str) -> str:
    return {'ns': 'nanosecond', 'us': 'microsecond', 'ms': 'millisecond', 's': 'second'}[u]