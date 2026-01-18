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
@pytest.mark.parametrize('src_encoding, dest_encoding', [('utf-8', 'ascii'), ('utf-8', 'latin-1')])
def test_transcoding_encoding_error(src_encoding, dest_encoding):
    stream = pa.transcoding_input_stream(pa.BufferReader('Ā'.encode(src_encoding)), src_encoding, dest_encoding)
    with pytest.raises(UnicodeEncodeError):
        stream.read(1)