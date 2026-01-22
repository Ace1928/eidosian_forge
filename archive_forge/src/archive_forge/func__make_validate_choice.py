import locale
import logging
import os
import pprint
import re
import sys
import warnings
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Dict
from typing_extensions import Literal
import numpy as np
def _make_validate_choice(accepted_values, allow_none=False, typeof=str):
    """Validate value is in accepted_values.

    Parameters
    ----------
    accepted_values : iterable
        Iterable containing all accepted_values.
    allow_none: boolean, optional
        Whether to accept ``None`` in addition to the values in ``accepted_values``.
    typeof: type, optional
        Type the values should be converted to.
    """

    def validate_choice(value):
        if allow_none and (value is None or (isinstance(value, str) and value.lower() == 'none')):
            return None
        try:
            value = typeof(value)
        except (ValueError, TypeError) as err:
            raise ValueError(f'Could not convert to {typeof.__name__}') from err
        if isinstance(value, str):
            value = value.lower()
        if value in accepted_values:
            value = {'true': True, 'false': False}.get(value, value)
            return value
        raise ValueError(f'{value} is not one of {accepted_values}{(' nor None' if allow_none else '')}')
    return validate_choice