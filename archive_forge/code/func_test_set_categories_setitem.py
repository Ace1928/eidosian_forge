import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import CategoricalAccessor
from pandas.core.indexes.accessors import Properties
def test_set_categories_setitem(self):
    df = DataFrame({'Survived': [1, 0, 1], 'Sex': [0, 1, 1]}, dtype='category')
    df['Survived'] = df['Survived'].cat.rename_categories(['No', 'Yes'])
    df['Sex'] = df['Sex'].cat.rename_categories(['female', 'male'])
    assert list(df['Sex']) == ['female', 'male', 'male']
    assert list(df['Survived']) == ['Yes', 'No', 'Yes']
    df['Sex'] = Categorical(df['Sex'], categories=['female', 'male'], ordered=False)
    df['Survived'] = Categorical(df['Survived'], categories=['No', 'Yes'], ordered=False)
    assert list(df['Sex']) == ['female', 'male', 'male']
    assert list(df['Survived']) == ['Yes', 'No', 'Yes']