import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
@doc(_doc_io_method_template, source='Arrow Table', params='at : pyarrow.Table', method='utils.from_arrow')
def _from_arrow(cls, at):
    return cls.io_cls.from_arrow(at)