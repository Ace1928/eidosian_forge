from gitdb.test.lib import (
from gitdb import (
from gitdb.util import hex_to_bin
import zlib
from gitdb.typ import (
import tempfile
import os
from io import BytesIO
def test_sha_writer(self):
    writer = Sha1Writer()
    assert 2 == writer.write(b'hi')
    assert len(writer.sha(as_hex=1)) == 40
    assert len(writer.sha(as_hex=0)) == 20
    prev_sha = writer.sha()
    writer.write(b'hi again')
    assert writer.sha() != prev_sha