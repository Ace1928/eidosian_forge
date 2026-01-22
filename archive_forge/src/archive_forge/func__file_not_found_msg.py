import os
import fsspec
import numpy as np
from pandas.io.common import is_fsspec_url, is_url
from modin.config import AsyncReadMode
from modin.logging import ClassLogger
from modin.utils import ModinAssumptionError
@classmethod
def _file_not_found_msg(cls, filename: str):
    return f"No such file: '{filename}'"