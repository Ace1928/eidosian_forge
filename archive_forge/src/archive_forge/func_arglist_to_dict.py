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
def arglist_to_dict(arglist: List[str]) -> Dict[str, str]:
    """Convert a list of arguments like ['arg1=val1', 'arg2=val2', ...] to a
    dict
    """
    return dict((x.split('=', 1) for x in arglist))