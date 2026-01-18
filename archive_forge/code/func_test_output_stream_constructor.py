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
def test_output_stream_constructor(tmpdir):
    if not Codec.is_available('gzip'):
        pytest.skip('gzip support is not built')
    with pa.CompressedOutputStream(tmpdir / 'ctor.gz', 'gzip') as stream:
        stream.write(b'test')
    with (tmpdir / 'ctor2.gz').open('wb') as f:
        with pa.CompressedOutputStream(f, 'gzip') as stream:
            stream.write(b'test')