import pytest
from jeepney.low_level import *
def test_parser_chunks():
    p = Parser()
    chunked = list(chunks(HELLO_METHOD_CALL, 16))
    for c in chunked[:-1]:
        assert p.feed(c) == []
    msg = p.feed(chunked[-1])[0]
    assert msg.header.fields[HeaderFields.member] == 'Hello'