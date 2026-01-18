import bz2
from contextlib import contextmanager
from io import (BytesIO, StringIO, TextIOWrapper, BufferedIOBase, IOBase)
import itertools
import gc
import gzip
import math
import os
import pathlib
import pytest
import sys
import tempfile
import weakref
import numpy as np
from pyarrow.util import guid
from pyarrow import Codec
import pyarrow as pa
@pytest.mark.parametrize('src_encoding, dest_encoding', [('utf-8', 'utf-16'), ('utf-16', 'utf-8'), ('utf-8', 'utf-32-le'), ('utf-8', 'utf-32-be')])
def test_transcoding_input_stream(src_encoding, dest_encoding):
    check_transcoding(unicode_transcoding_example, src_encoding, dest_encoding, [1000, 0])
    check_transcoding(unicode_transcoding_example, src_encoding, dest_encoding, itertools.cycle([1, 2, 3, 5]))