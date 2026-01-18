import functools
import warnings
from functools import update_wrapper
import joblib
from .._config import config_context, get_config
def with_config(self, config):
    self.config = config
    return self