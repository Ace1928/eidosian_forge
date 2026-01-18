import pytest
from pathlib import Path
import datetime
from mock import patch
import numpy
from .._msgpack_api import read_msgpack, write_msgpack
from .._msgpack_api import msgpack_loads, msgpack_dumps
from .._msgpack_api import msgpack_encoders, msgpack_decoders
from .util import make_tempdir
def test_read_msgpack_file():
    file_contents = b'\x81\xa5hello\xa5world'
    with make_tempdir({'tmp.msg': file_contents}, mode='wb') as temp_dir:
        file_path = temp_dir / 'tmp.msg'
        assert file_path.exists()
        data = read_msgpack(file_path)
    assert len(data) == 1
    assert data['hello'] == 'world'