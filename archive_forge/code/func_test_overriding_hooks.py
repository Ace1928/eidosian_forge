import array
from srsly import msgpack
from srsly.msgpack._ext_type import ExtType
def test_overriding_hooks():

    def default(obj):
        if isinstance(obj, int):
            return {'__type__': 'long', '__data__': str(obj)}
        else:
            return obj
    obj = {'testval': int(1823746192837461928374619)}
    refobj = {'testval': default(obj['testval'])}
    refout = msgpack.packb(refobj)
    assert isinstance(refout, (str, bytes))
    testout = msgpack.packb(obj, default=default)
    assert refout == testout