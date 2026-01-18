import collections
import dataclasses
import importlib.metadata
import inspect
import logging
import multiprocessing
import os
import sys
import traceback
import types
import warnings
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple
from importlib.machinery import ModuleSpec
from unittest import mock
import duet
import numpy as np
import pandas as pd
import pytest
import sympy
from _pytest.outcomes import Failed
import cirq.testing
from cirq._compat import (
def test_same_name_submodule_earlier_in_subtree():
    """Tests whether module resolution works in the right order.

    We have two packages with a bool `DUPE_CONSTANT` attribute each:
       1. cirq.testing._compat_test_data.module_a.sub.dupe.DUPE_CONSTANT=True # the right one
       2. cirq.testing._compat_test_data.module_a.dupe.DUPE_CONSTANT=False # the wrong one

    If the new module's (in this case cirq.testing._compat_test_data.module_a) path has precedence
    during module spec resolution, dupe number 2 is going to get resolved.

    You might wonder where this comes up in cirq. There was a bug where the lookup path was not in
    the right order. The motivating example is cirq.ops.calibration vs the
    cirq.ops.engine.calibration packages. The wrong resolution resulted in false circular
    imports!
    """
    subprocess_context(_test_same_name_submodule_earlier_in_subtree_inner)()