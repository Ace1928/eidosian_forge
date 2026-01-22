import rpy2.robjects.conversion as conversion
import rpy2.rinterface as rinterface
from rpy2.rinterface_lib import na_values
from rpy2.rinterface import IntSexpVector
from rpy2.rinterface import ListSexpVector
from rpy2.rinterface import SexpVector
from rpy2.rinterface import StrSexpVector
import datetime
import functools
import math
import numpy  # type: ignore
import pandas  # type: ignore
import pandas.core.series  # type: ignore
from pandas.core.frame import DataFrame as PandasDataFrame  # type: ignore
from pandas.core.dtypes.api import is_datetime64_any_dtype  # type: ignore
import warnings
from collections import OrderedDict
from rpy2.robjects.vectors import (BoolVector,
import rpy2.robjects.numpy2ri as numpy2ri
def _to_pandas_factor(obj):
    codes = [x - 1 if x > 0 else -1 for x in numpy.array(obj)]
    res = pandas.Categorical.from_codes(codes, categories=list(obj.do_slot('levels')), ordered='ordered' in obj.rclass)
    return res