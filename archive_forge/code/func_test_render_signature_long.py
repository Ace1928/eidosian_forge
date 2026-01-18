from contextlib import contextmanager
from inspect import signature, Signature, Parameter
import inspect
import os
import pytest
import re
import sys
from .. import oinspect
from decorator import decorator
from IPython.testing.tools import AssertPrints, AssertNotPrints
from IPython.utils.path import compress_user
def test_render_signature_long():
    from typing import Optional

    def long_function(a_really_long_parameter: int, and_another_long_one: bool=False, let_us_make_sure_this_is_looong: Optional[str]=None) -> bool:
        pass
    sig = oinspect._render_signature(signature(long_function), long_function.__name__)
    expected = 'long_function(\n    a_really_long_parameter: int,\n    and_another_long_one: bool = False,\n    let_us_make_sure_this_is_looong: Optional[str] = None,\n) -> bool'
    assert sig == expected