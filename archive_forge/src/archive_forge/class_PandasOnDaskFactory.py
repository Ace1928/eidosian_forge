import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@doc(_doc_factory_class, execution_name='PandasOnDask')
class PandasOnDaskFactory(BaseFactory):

    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name='``PandasOnDaskIO``')
    def prepare(cls):
        from modin.core.execution.dask.implementations.pandas_on_dask.io import PandasOnDaskIO
        cls.io_cls = PandasOnDaskIO