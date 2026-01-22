from __future__ import annotations
import os
import pickle
import time
from typing import TYPE_CHECKING
from fsspec.utils import atomic_write
Update metadata for specific file in memory, do not save