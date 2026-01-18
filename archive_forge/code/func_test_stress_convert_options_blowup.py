import abc
import bz2
from datetime import date, datetime
from decimal import Decimal
import gc
import gzip
import io
import itertools
import os
import select
import shutil
import signal
import string
import tempfile
import threading
import time
import unittest
import weakref
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.csv import (
from pyarrow.tests import util
def test_stress_convert_options_blowup(self):
    try:
        clock = time.thread_time
    except AttributeError:
        clock = time.time
    num_columns = 10000
    col_names = ['K{}'.format(i) for i in range(num_columns)]
    csv = make_empty_csv(col_names)
    t1 = clock()
    convert_options = ConvertOptions(column_types={k: pa.string() for k in col_names[::2]})
    table = self.read_bytes(csv, convert_options=convert_options)
    dt = clock() - t1
    assert dt <= 10.0
    assert table.num_columns == num_columns
    assert table.num_rows == 0
    assert table.column_names == col_names