import copy
import json
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from packaging import version
from . import __version__
from .dynamic_module_utils import custom_object_save
from .utils import (
@staticmethod
def _set_token_in_kwargs(kwargs, token=None):
    """Temporary method to deal with `token` and `use_auth_token`.

        This method is to avoid apply the same changes in all model config classes that overwrite `from_pretrained`.

        Need to clean up `use_auth_token` in a follow PR.
        """
    if token is None:
        token = kwargs.pop('token', None)
    use_auth_token = kwargs.pop('use_auth_token', None)
    if use_auth_token is not None:
        warnings.warn('The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.', FutureWarning)
        if token is not None:
            raise ValueError('`token` and `use_auth_token` are both specified. Please set only the argument `token`.')
        token = use_auth_token
    if token is not None:
        kwargs['token'] = token