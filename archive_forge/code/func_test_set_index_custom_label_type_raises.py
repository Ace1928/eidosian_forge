from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
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