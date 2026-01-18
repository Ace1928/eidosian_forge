import matplotlib
import numpy as np
import pandas
import pytest
from pandas.core.dtypes.common import is_list_like
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('engine', ['python', 'numexpr'])
def test_query_with_element_access_issue_4580(engine):
    pdf = pandas.DataFrame({'a': [0, 1, 2]})
    df = pd.concat([pd.DataFrame(pdf[:1]), pd.DataFrame(pdf[1:])])
    eval_general(df, pdf, lambda df: df.query('a == a[0]', engine=engine))