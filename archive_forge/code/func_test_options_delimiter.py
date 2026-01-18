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
def test_options_delimiter(self):
    rows = b'a;b,c\nde,fg;eh\n'
    reader = self.open_bytes(rows)
    expected_schema = pa.schema([('a;b', pa.string()), ('c', pa.string())])
    self.check_reader(reader, expected_schema, [{'a;b': ['de'], 'c': ['fg;eh']}])
    opts = ParseOptions(delimiter=';')
    reader = self.open_bytes(rows, parse_options=opts)
    expected_schema = pa.schema([('a', pa.string()), ('b,c', pa.string())])
    self.check_reader(reader, expected_schema, [{'a': ['de,fg'], 'b,c': ['eh']}])