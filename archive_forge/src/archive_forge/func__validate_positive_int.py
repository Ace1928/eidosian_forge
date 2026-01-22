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
def _validate_positive_int(value):
    """Validate value is a natural number."""
    try:
        value = int(value)
    except ValueError as err:
        raise ValueError('Could not convert to int') from err
    if value > 0:
        return value
    else:
        raise ValueError('Only positive values are valid')