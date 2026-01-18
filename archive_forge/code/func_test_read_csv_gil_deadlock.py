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
def test_read_csv_gil_deadlock():
    data = b'a,b,c'

    class MyBytesIO(io.BytesIO):

        def read(self, *args):
            time.sleep(0.001)
            return super().read(*args)

        def readinto(self, *args):
            time.sleep(0.001)
            return super().readinto(*args)
    for i in range(20):
        with pytest.raises(pa.ArrowInvalid):
            read_csv(MyBytesIO(data))