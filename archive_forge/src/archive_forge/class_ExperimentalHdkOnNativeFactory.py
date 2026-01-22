import re
import typing
import warnings
import pandas
from pandas.util._decorators import doc
from modin.config import IsExperimental
from modin.core.io import BaseIO
from modin.utils import get_current_execution
@doc(_doc_factory_class, execution_name='experimental HdkOnNative')
class ExperimentalHdkOnNativeFactory(BaseFactory):

    @classmethod
    @doc(_doc_factory_prepare_method, io_module_name='experimental ``HdkOnNativeIO``')
    def prepare(cls):
        from modin.experimental.core.execution.native.implementations.hdk_on_native.io import HdkOnNativeIO
        if not IsExperimental.get():
            raise ValueError("'HdkOnNative' only works in experimental mode.")
        cls.io_cls = HdkOnNativeIO