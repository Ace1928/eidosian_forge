from pytest import raises
import datetime
from srsly.msgpack import packb, unpackb, Unpacker, FormatError, StackError, OutOfData
def test_raise_from_object_hook():

    def hook(obj):
        raise DummyException
    raises(DummyException, unpackb, packb({}), object_hook=hook)
    raises(DummyException, unpackb, packb({'fizz': 'buzz'}), object_hook=hook)
    raises(DummyException, unpackb, packb({'fizz': 'buzz'}), object_pairs_hook=hook)
    raises(DummyException, unpackb, packb({'fizz': {'buzz': 'spam'}}), object_hook=hook)
    raises(DummyException, unpackb, packb({'fizz': {'buzz': 'spam'}}), object_pairs_hook=hook)