import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
@doc(_doc_io_method_template, source='an HDFStore', params=_doc_io_method_kwargs_params, method='read_hdf')
def _read_hdf(cls, **kwargs):
    return cls.io_cls.read_hdf(**kwargs)