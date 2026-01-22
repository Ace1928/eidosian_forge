from __future__ import annotations
from .common import CMakeException, CMakeBuildFile, CMakeConfiguration
import typing as T
from .. import mlog
from pathlib import Path
import json
import re
def _strip_data(self, data: T.Any) -> T.Any:
    if isinstance(data, list):
        for idx, i in enumerate(data):
            data[idx] = self._strip_data(i)
    elif isinstance(data, dict):
        new = {}
        for key, val in data.items():
            if key not in STRIP_KEYS:
                new[key] = self._strip_data(val)
        data = new
    return data