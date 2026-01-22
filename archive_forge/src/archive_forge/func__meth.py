from __future__ import annotations
import re
import typing
from typing import Any, Callable, TypeVar
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import _api
def _meth(self, arg: T) -> T:
    return arg