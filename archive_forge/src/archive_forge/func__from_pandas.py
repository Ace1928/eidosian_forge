import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
@doc(_doc_io_method_template, source='pandas DataFrame', params='df : pandas.DataFrame', method='utils.from_pandas')
def _from_pandas(cls, df):
    return cls.io_cls.from_pandas(df)