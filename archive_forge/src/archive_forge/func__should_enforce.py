import os
import warnings
from datetime import date
from inspect import cleandoc
from textwrap import indent
from typing import Optional, Tuple
def _should_enforce():
    enforce = os.getenv('SETUPTOOLS_ENFORCE_DEPRECATION', 'false').lower()
    return enforce in ('true', 'on', 'ok', '1')