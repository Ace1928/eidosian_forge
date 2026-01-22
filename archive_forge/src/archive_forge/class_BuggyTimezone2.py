from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
class BuggyTimezone2(datetime.tzinfo):
    """
        Wrong offset type
        """

    def tzname(self, dt):
        return None

    def utcoffset(self, dt):
        return 'one hour'