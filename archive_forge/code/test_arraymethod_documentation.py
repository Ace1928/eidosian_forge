from __future__ import annotations
import sys
import types
from typing import Any
import pytest
import numpy as np
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
Test `ndarray.__class_getitem__`.