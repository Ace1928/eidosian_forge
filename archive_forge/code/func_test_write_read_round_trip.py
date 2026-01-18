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
def test_write_read_round_trip():
    t = pa.Table.from_arrays([[1, 2, 3], ['a', 'b', 'c']], ['c1', 'c2'])
    record_batch = t.to_batches(max_chunksize=4)[0]
    for data in [t, record_batch]:
        buf = io.BytesIO()
        write_csv(data, buf, WriteOptions(include_header=True))
        buf.seek(0)
        assert t == read_csv(buf)
        buf = io.BytesIO()
        write_csv(data, buf, WriteOptions(include_header=False))
        buf.seek(0)
        read_options = ReadOptions(column_names=t.column_names)
        assert t == read_csv(buf, read_options=read_options)
    for read_options, parse_options, write_options in [(None, None, WriteOptions(include_header=True)), (ReadOptions(column_names=t.column_names), None, WriteOptions(include_header=False)), (None, ParseOptions(delimiter='|'), WriteOptions(include_header=True, delimiter='|')), (ReadOptions(column_names=t.column_names), ParseOptions(delimiter='\t'), WriteOptions(include_header=False, delimiter='\t'))]:
        buf = io.BytesIO()
        with CSVWriter(buf, t.schema, write_options=write_options) as writer:
            writer.write_table(t)
        buf.seek(0)
        assert t == read_csv(buf, read_options=read_options, parse_options=parse_options)
        buf = io.BytesIO()
        with CSVWriter(buf, t.schema, write_options=write_options) as writer:
            for batch in t.to_batches(max_chunksize=1):
                writer.write_batch(batch)
        buf.seek(0)
        assert t == read_csv(buf, read_options=read_options, parse_options=parse_options)