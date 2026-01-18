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
def test_read_csv_does_not_close_passed_file_handles():
    buf = io.BytesIO(b'a,b,c\n1,2,3\n4,5,6')
    read_csv(buf)
    assert not buf.closed