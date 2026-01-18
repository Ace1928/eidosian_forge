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
def validate_with_none(value):
    if value is None or (isinstance(value, str) and value.lower() == 'none'):
        return None
    return base_validator(value)