from wsme.exc import (ClientSideError, InvalidInput, MissingArgument,
def test_unknownargument():
    e = UnknownArgument('argname', 'error message')
    assert e.faultstring == 'Unknown argument: "argname": error message', e.faultstring