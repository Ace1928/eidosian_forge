from __future__ import annotations
import contextlib
import warnings
import numpy as np
import pandas as pd
import pytest
from xarray import (
from xarray.backends.common import WritableCFDataStore
from xarray.backends.memory import InMemoryDataStore
from xarray.conventions import decode_cf
from xarray.testing import assert_identical
from xarray.tests import (
from xarray.tests.test_backends import CFEncodedBase
class CFEncodedInMemoryStore(WritableCFDataStore, InMemoryDataStore):

    def encode_variable(self, var):
        """encode one variable"""
        coder = coding.strings.EncodedStringCoder(allows_unicode=True)
        var = coder.encode(var)
        return var