from __future__ import print_function, unicode_literals
import sys
import pytest  # NOQA
import warnings  # NOQA
from pathlib import Path

    version indication, return True if version matches.
    match should be something like 3.6+, or [2.7, 3.3] etc. Floats
    are converted to strings. Single values are made into lists.
    