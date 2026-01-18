from textwrap import dedent
import pytest
from pandas import (
from pandas.io.formats.style import Styler
def test_set_td_classes_non_unique_raises(styler):
    classes = DataFrame([['1', '2'], ['3', '4']], columns=['c', 'd'], index=['a', 'b'])
    styler.set_td_classes(classes=classes)
    classes = DataFrame([['1', '2'], ['3', '4']], columns=['c', 'c'], index=['a', 'b'])
    with pytest.raises(KeyError, match='Classes render only if `classes` has unique'):
        styler.set_td_classes(classes=classes)
    classes = DataFrame([['1', '2'], ['3', '4']], columns=['c', 'd'], index=['a', 'a'])
    with pytest.raises(KeyError, match='Classes render only if `classes` has unique'):
        styler.set_td_classes(classes=classes)