from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestSetIndexCustomLabelType:

    def test_set_index_custom_label_type(self):

        class Thing:

            def __init__(self, name, color) -> None:
                self.name = name
                self.color = color

            def __str__(self) -> str:
                return f'<Thing {repr(self.name)}>'
            __repr__ = __str__
        thing1 = Thing('One', 'red')
        thing2 = Thing('Two', 'blue')
        df = DataFrame({thing1: [0, 1], thing2: [2, 3]})
        expected = DataFrame({thing1: [0, 1]}, index=Index([2, 3], name=thing2))
        result = df.set_index(thing2)
        tm.assert_frame_equal(result, expected)
        result = df.set_index([thing2])
        tm.assert_frame_equal(result, expected)
        thing3 = Thing('Three', 'pink')
        msg = "<Thing 'Three'>"
        with pytest.raises(KeyError, match=msg):
            df.set_index(thing3)
        with pytest.raises(KeyError, match=msg):
            df.set_index([thing3])

    def test_set_index_custom_label_hashable_iterable(self):

        class Thing(frozenset):

            def __repr__(self) -> str:
                tmp = sorted(self)
                joined_reprs = ', '.join(map(repr, tmp))
                return f'frozenset({{{joined_reprs}}})'
        thing1 = Thing(['One', 'red'])
        thing2 = Thing(['Two', 'blue'])
        df = DataFrame({thing1: [0, 1], thing2: [2, 3]})
        expected = DataFrame({thing1: [0, 1]}, index=Index([2, 3], name=thing2))
        result = df.set_index(thing2)
        tm.assert_frame_equal(result, expected)
        result = df.set_index([thing2])
        tm.assert_frame_equal(result, expected)
        thing3 = Thing(['Three', 'pink'])
        msg = "frozenset\\(\\{'Three', 'pink'\\}\\)"
        with pytest.raises(KeyError, match=msg):
            df.set_index(thing3)
        with pytest.raises(KeyError, match=msg):
            df.set_index([thing3])

    def test_set_index_custom_label_type_raises(self):

        class Thing(set):

            def __init__(self, name, color) -> None:
                self.name = name
                self.color = color

            def __str__(self) -> str:
                return f'<Thing {repr(self.name)}>'
        thing1 = Thing('One', 'red')
        thing2 = Thing('Two', 'blue')
        df = DataFrame([[0, 2], [1, 3]], columns=[thing1, thing2])
        msg = 'The parameter "keys" may be a column key, .*'
        with pytest.raises(TypeError, match=msg):
            df.set_index(thing2)
        with pytest.raises(TypeError, match=msg):
            df.set_index([thing2])

    def test_set_index_periodindex(self):
        df = DataFrame(np.random.default_rng(2).random(6))
        idx1 = period_range('2011/01/01', periods=6, freq='M')
        idx2 = period_range('2013', periods=6, freq='Y')
        df = df.set_index(idx1)
        tm.assert_index_equal(df.index, idx1)
        df = df.set_index(idx2)
        tm.assert_index_equal(df.index, idx2)