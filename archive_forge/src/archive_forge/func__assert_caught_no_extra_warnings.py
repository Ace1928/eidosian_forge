from __future__ import annotations
from contextlib import (
import inspect
import re
import sys
from typing import (
import warnings
from pandas.compat import PY311
def _assert_caught_no_extra_warnings(*, caught_warnings: Sequence[warnings.WarningMessage], expected_warning: type[Warning] | bool | tuple[type[Warning], ...] | None) -> None:
    """Assert that no extra warnings apart from the expected ones are caught."""
    extra_warnings = []
    for actual_warning in caught_warnings:
        if _is_unexpected_warning(actual_warning, expected_warning):
            if actual_warning.category == ResourceWarning:
                if 'unclosed <ssl.SSLSocket' in str(actual_warning.message):
                    continue
                if any(('matplotlib' in mod for mod in sys.modules)):
                    continue
            if PY311 and actual_warning.category == EncodingWarning:
                continue
            extra_warnings.append((actual_warning.category.__name__, actual_warning.message, actual_warning.filename, actual_warning.lineno))
    if extra_warnings:
        raise AssertionError(f'Caused unexpected warning(s): {repr(extra_warnings)}')