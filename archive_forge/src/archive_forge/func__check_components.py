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
def _check_components(complist: Collection[Any]) -> None:
    if len({convert(c) for c in complist}) != len(complist):
        raise ValueError(f'Some paths in {complist!r} convert to the same object, please update your settings')