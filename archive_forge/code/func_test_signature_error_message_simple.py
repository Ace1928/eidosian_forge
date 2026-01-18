import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock
import numpy as np
from numpy.testing import (
from numpy.core.overrides import (
from numpy.compat import pickle
import pytest
@pytest.mark.parametrize('name', ['concatenate', 'mean', 'asarray'])
def test_signature_error_message_simple(self, name):
    func = getattr(np, name)
    try:
        func()
    except TypeError as e:
        exc = e
    assert exc.args[0].startswith(f'{name}()')