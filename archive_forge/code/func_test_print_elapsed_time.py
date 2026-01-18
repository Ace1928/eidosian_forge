import string
import timeit
import warnings
from copy import copy
from itertools import chain
from unittest import SkipTest
import numpy as np
import pytest
from sklearn import config_context
from sklearn.externals._packaging.version import parse as parse_version
from sklearn.utils import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize(['message', 'expected'], [('hello', _message_with_time('ABC', 'hello', 0.1) + '\n'), ('', _message_with_time('ABC', '', 0.1) + '\n'), (None, '')])
def test_print_elapsed_time(message, expected, capsys, monkeypatch):
    monkeypatch.setattr(timeit, 'default_timer', lambda: 0)
    with _print_elapsed_time('ABC', message):
        monkeypatch.setattr(timeit, 'default_timer', lambda: 0.1)
    assert capsys.readouterr().out == expected