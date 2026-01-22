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
def _find_spec_deprecated_multiple_times():
    """to ensure the idempotency of the aliasing loader change"""
    import importlib.util
    assert importlib.util.find_spec('cirq.testing._compat_test_data.fake_a')
    assert importlib.util.find_spec('cirq.testing._compat_test_data.fake_a')