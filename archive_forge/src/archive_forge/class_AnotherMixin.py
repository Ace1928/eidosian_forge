import importlib
from collections import namedtuple
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
from sklearn.utils.fixes import CSR_CONTAINERS
class AnotherMixin:

    def __init_subclass__(cls, custom_parameter, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.custom_parameter = custom_parameter