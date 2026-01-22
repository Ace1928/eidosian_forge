import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@classmethod
@doc(_doc_io_method_raw_template, source='SQL files', params=_doc_io_method_kwargs_params)
def _read_sql_distributed(cls, **kwargs):
    current_execution = get_current_execution()
    if current_execution not in supported_executions:
        extra_parameters = ('partition_column', 'lower_bound', 'upper_bound', 'max_sessions')
        if any((param in kwargs and kwargs[param] is not None for param in extra_parameters)):
            warnings.warn(f'Distributed read_sql() was only implemented for {', '.join(supported_executions)} executions.')
        for param in extra_parameters:
            del kwargs[param]
        return cls.io_cls.read_sql(**kwargs)
    return cls.io_cls.read_sql_distributed(**kwargs)