import copy
import logging
from s3transfer.utils import get_callbacks
def _get_kwargs_with_params_to_exclude(self, kwargs, exclude):
    filtered_kwargs = {}
    for param, value in kwargs.items():
        if param in exclude:
            continue
        filtered_kwargs[param] = value
    return filtered_kwargs