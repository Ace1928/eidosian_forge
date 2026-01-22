from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
class EqualsTests:

    def test_not_equals_numeric(self, index):
        assert not index.equals(Index(index.asi8))
        assert not index.equals(Index(index.asi8.astype('u8')))
        assert not index.equals(Index(index.asi8).astype('f8'))

    def test_equals(self, index):
        assert index.equals(index)
        assert index.equals(index.astype(object))
        assert index.equals(CategoricalIndex(index))
        assert index.equals(CategoricalIndex(index.astype(object)))

    def test_not_equals_non_arraylike(self, index):
        assert not index.equals(list(index))

    def test_not_equals_strings(self, index):
        other = Index([str(x) for x in index], dtype=object)
        assert not index.equals(other)
        assert not index.equals(CategoricalIndex(other))

    def test_not_equals_misc_strs(self, index):
        other = Index(list('abc'))
        assert not index.equals(other)