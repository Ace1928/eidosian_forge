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
def test_read_options(pickle_module):
    cls = ReadOptions
    opts = cls()
    check_options_class(cls, use_threads=[True, False], skip_rows=[0, 3], column_names=[[], ['ab', 'cd']], autogenerate_column_names=[False, True], encoding=['utf8', 'utf16'], skip_rows_after_names=[0, 27])
    check_options_class_pickling(cls, pickler=pickle_module, use_threads=True, skip_rows=3, column_names=['ab', 'cd'], autogenerate_column_names=False, encoding='utf16', skip_rows_after_names=27)
    assert opts.block_size > 0
    opts.block_size = 12345
    assert opts.block_size == 12345
    opts = cls(block_size=1234)
    assert opts.block_size == 1234
    opts.validate()
    match = 'ReadOptions: block_size must be at least 1: 0'
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.block_size = 0
        opts.validate()
    match = 'ReadOptions: skip_rows cannot be negative: -1'
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.skip_rows = -1
        opts.validate()
    match = 'ReadOptions: skip_rows_after_names cannot be negative: -1'
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.skip_rows_after_names = -1
        opts.validate()
    match = 'ReadOptions: autogenerate_column_names cannot be true when column_names are provided'
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.autogenerate_column_names = True
        opts.column_names = ('a', 'b')
        opts.validate()