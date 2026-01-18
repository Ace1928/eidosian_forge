import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util

    Check we can call `MapScalar.as_py` with custom field names

    See https://github.com/apache/arrow/issues/36809
    