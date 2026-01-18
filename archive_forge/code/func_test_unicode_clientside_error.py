from wsme.exc import (ClientSideError, InvalidInput, MissingArgument,
def test_unicode_clientside_error():
    e = ClientSideError('ファシリ')
    assert e.faultstring == 'ファシリ'