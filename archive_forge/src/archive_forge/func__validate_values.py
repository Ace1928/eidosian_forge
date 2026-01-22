import numbers
import os
import sys
import warnings
from configparser import ConfigParser
from operator import itemgetter
from pathlib import Path
from typing import (
from scrapy.exceptions import ScrapyDeprecationWarning, UsageError
from scrapy.settings import BaseSettings
from scrapy.utils.deprecate import update_classpath
from scrapy.utils.python import without_none_values
def _validate_values(compdict: Mapping[Any, Any]) -> None:
    """Fail if a value in the components dict is not a real number or None."""
    for name, value in compdict.items():
        if value is not None and (not isinstance(value, numbers.Real)):
            raise ValueError(f'Invalid value {value} for component {name}, please provide a real number or None instead')