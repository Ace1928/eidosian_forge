import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
@staticmethod
def _convert_categorical(from_frame: DataFrame) -> DataFrame:
    """
        Emulate the categorical casting behavior we expect from roundtripping.
        """
    for col in from_frame:
        ser = from_frame[col]
        if isinstance(ser.dtype, CategoricalDtype):
            cat = ser._values.remove_unused_categories()
            if cat.categories.dtype == object:
                categories = pd.Index._with_infer(cat.categories._values)
                cat = cat.set_categories(categories)
            from_frame[col] = cat
    return from_frame