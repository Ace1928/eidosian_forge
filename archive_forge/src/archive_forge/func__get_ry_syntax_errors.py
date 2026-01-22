from itertools import chain, repeat
from textwrap import dedent, indent
from typing import TYPE_CHECKING
from unittest import TestCase
import pytest
from IPython.core.async_helpers import _should_be_async
from IPython.testing.decorators import skip_without
def _get_ry_syntax_errors(self):
    test_cases = []
    test_cases.append(('class', dedent('\n        class V:\n            {val}\n        ')))
    test_cases.append(('nested-class', dedent('\n        class V:\n            class C:\n                {val}\n        ')))
    return test_cases