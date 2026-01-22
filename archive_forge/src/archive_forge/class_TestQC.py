import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Optional
import boto3
import numpy as np
import pandas
import pytest
import requests
import s3fs
from pandas.util._decorators import doc
import modin.utils  # noqa: E402
import uuid  # noqa: E402
import modin  # noqa: E402
import modin.config  # noqa: E402
import modin.tests.config  # noqa: E402
from modin.config import (  # noqa: E402
from modin.core.execution.dispatching.factories import factories  # noqa: E402
from modin.core.execution.python.implementations.pandas_on_python.io import (  # noqa: E402
from modin.core.storage_formats import (  # noqa: E402
from modin.tests.pandas.utils import (  # noqa: E402
class TestQC(BaseQueryCompiler):

    def __init__(self, modin_frame):
        self._modin_frame = modin_frame

    def finalize(self):
        self._modin_frame.finalize()

    def execute(self):
        self.finalize()
        self._modin_frame.wait_computations()

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    @classmethod
    def from_arrow(cls, at, data_cls):
        return cls(data_cls.from_arrow(at))

    def free(self):
        pass

    def to_dataframe(self, nan_as_null: bool=False, allow_copy: bool=True):
        raise NotImplementedError('The selected execution does not implement the DataFrame exchange protocol.')

    @classmethod
    def from_dataframe(cls, df, data_cls):
        raise NotImplementedError('The selected execution does not implement the DataFrame exchange protocol.')
    to_pandas = PandasQueryCompiler.to_pandas
    default_to_pandas = PandasQueryCompiler.default_to_pandas