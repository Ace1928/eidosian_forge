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
def test_small_random_csv(self):
    csv, expected = make_random_csv(num_cols=2, num_rows=10)
    reader = self.open_bytes(csv)
    table = reader.read_all()
    assert table.schema == expected.schema
    assert table.equals(expected)
    assert table.to_pydict() == expected.to_pydict()