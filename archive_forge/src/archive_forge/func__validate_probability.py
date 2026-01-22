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
def _validate_probability(value):
    """Validate a probability: a float between 0 and 1."""
    value = _validate_float(value)
    if value < 0 or value > 1:
        raise ValueError('Only values between 0 and 1 are valid.')
    return value