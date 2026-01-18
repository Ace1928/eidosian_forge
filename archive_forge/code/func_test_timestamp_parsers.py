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
def test_timestamp_parsers(self):
    rows = b'a,b\n1970/01/01,1980-01-01 00\n1970/01/02,1980-01-02 00\n'
    opts = ConvertOptions()
    table = self.read_bytes(rows, convert_options=opts)
    schema = pa.schema([('a', pa.string()), ('b', pa.timestamp('s'))])
    assert table.schema == schema
    assert table.to_pydict() == {'a': ['1970/01/01', '1970/01/02'], 'b': [datetime(1980, 1, 1), datetime(1980, 1, 2)]}
    opts.timestamp_parsers = ['%Y/%m/%d']
    table = self.read_bytes(rows, convert_options=opts)
    schema = pa.schema([('a', pa.timestamp('s')), ('b', pa.string())])
    assert table.schema == schema
    assert table.to_pydict() == {'a': [datetime(1970, 1, 1), datetime(1970, 1, 2)], 'b': ['1980-01-01 00', '1980-01-02 00']}
    opts.timestamp_parsers = ['%Y/%m/%d', ISO8601]
    table = self.read_bytes(rows, convert_options=opts)
    schema = pa.schema([('a', pa.timestamp('s')), ('b', pa.timestamp('s'))])
    assert table.schema == schema
    assert table.to_pydict() == {'a': [datetime(1970, 1, 1), datetime(1970, 1, 2)], 'b': [datetime(1980, 1, 1), datetime(1980, 1, 2)]}