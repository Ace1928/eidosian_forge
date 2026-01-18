import os
from io import BytesIO
import pytest
from nltk.corpus.reader import SeekableUnicodeStreamReader
def test_reader_stream_closes_when_deleted():
    reader = SeekableUnicodeStreamReader(BytesIO(b''), 'ascii')
    assert not reader.stream.closed
    reader.__del__()
    assert reader.stream.closed