from io import BytesIO
import sys
import pytest
from srsly.msgpack import Unpacker, packb, OutOfData, ExtType
def test_unpacker_ext_hook():

    class MyUnpacker(Unpacker):

        def __init__(self):
            super(MyUnpacker, self).__init__(ext_hook=self._hook, raw=False)

        def _hook(self, code, data):
            if code == 1:
                return int(data)
            else:
                return ExtType(code, data)
    unpacker = MyUnpacker()
    unpacker.feed(packb({'a': 1}))
    assert unpacker.unpack() == {'a': 1}
    unpacker.feed(packb({'a': ExtType(1, b'123')}))
    assert unpacker.unpack() == {'a': 123}
    unpacker.feed(packb({'a': ExtType(2, b'321')}))
    assert unpacker.unpack() == {'a': ExtType(2, b'321')}