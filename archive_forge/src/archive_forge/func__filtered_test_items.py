from __future__ import annotations
import functools
import glob
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
from collections import defaultdict
from typing import Any, Callable, Iterable, Pattern, Sequence
from urllib.request import urlretrieve
import numpy as np
import onnx
import onnx.reference
from onnx import ONNX_ML, ModelProto, NodeProto, TypeProto, ValueInfoProto, numpy_helper
from onnx.backend.base import Backend
from onnx.backend.test.case.test_case import TestCase
from onnx.backend.test.loader import load_model_tests
from onnx.backend.test.runner.item import TestItem
@property
def _filtered_test_items(self) -> dict[str, dict[str, TestItem]]:
    filtered: dict[str, dict[str, TestItem]] = {}
    for category, items_map in self._test_items.items():
        filtered[category] = {}
        for name, item in items_map.items():
            if self._include_patterns and (not any((include.search(name) for include in self._include_patterns))):
                item.func = unittest.skip('no matched include pattern')(item.func)
            for exclude in self._exclude_patterns:
                if exclude.search(name):
                    item.func = unittest.skip(f'matched exclude pattern "{exclude.pattern}"')(item.func)
            for xfail in self._xfail_patterns:
                if xfail.search(name):
                    item.func = unittest.expectedFailure(item.func)
            filtered[category][name] = item
    return filtered