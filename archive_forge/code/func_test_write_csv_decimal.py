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
@pytest.mark.parametrize('type_factory', (lambda: pa.decimal128(20, 1), lambda: pa.decimal128(38, 15), lambda: pa.decimal256(20, 1), lambda: pa.decimal256(76, 10)))
def test_write_csv_decimal(tmpdir, type_factory):
    type = type_factory()
    table = pa.table({'col': pa.array([1, 2]).cast(type)})
    write_csv(table, tmpdir / 'out.csv')
    out = read_csv(tmpdir / 'out.csv')
    assert out.column('col').cast(type) == table.column('col')