import inspect
import pprint
from collections import OrderedDict
from .._config import get_config
from ..base import BaseEstimator
from . import is_scalar_nan
Pretty printing for key-value tuples from dict or parameters.