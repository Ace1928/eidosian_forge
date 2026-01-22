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
class TestBZ2CSVRead(BaseTestCompressedCSVRead, unittest.TestCase):
    csv_filename = 'compressed.csv.bz2'

    def write_file(self, path, contents):
        with bz2.BZ2File(path, 'w') as f:
            f.write(contents)