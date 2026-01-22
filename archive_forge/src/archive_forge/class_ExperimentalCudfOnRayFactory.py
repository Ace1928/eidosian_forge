import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@doc(_doc_factory_class, execution_name='cuDFOnRay')
class ExperimentalCudfOnRayFactory(BaseFactory):

    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name='``cuDFOnRayIO``')
    def prepare(cls):
        from modin.core.execution.ray.implementations.cudf_on_ray.io import cuDFOnRayIO
        if not IsExperimental.get():
            raise ValueError("'CudfOnRay' only works in experimental mode.")
        cls.io_cls = cuDFOnRayIO