from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
def test_run_end_encode():
    check_run_end_encode_decode()
    check_run_end_encode_decode(pc.RunEndEncodeOptions(pa.int16()))
    check_run_end_encode_decode(pc.RunEndEncodeOptions('int32'))
    check_run_end_encode_decode(pc.RunEndEncodeOptions(pa.int64()))