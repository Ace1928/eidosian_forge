from __future__ import annotations
import inspect
import warnings
import pytest
from .._deprecate import (
from . import module_with_deprecations
def test_warn_deprecated_no_instead_or_issue(recwarn_always: pytest.WarningsRecorder) -> None:
    warn_deprecated('water', '1.3', issue=None, instead=None)
    assert len(recwarn_always) == 1
    got = recwarn_always.pop(TrioDeprecationWarning)
    assert isinstance(got.message, Warning)
    assert 'water is deprecated' in got.message.args[0]
    assert 'no replacement' in got.message.args[0]
    assert 'Trio 1.3' in got.message.args[0]