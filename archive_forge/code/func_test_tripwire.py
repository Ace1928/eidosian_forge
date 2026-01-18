import pytest
from ..tripwire import TripWire, TripWireError, is_tripwire
def test_tripwire():
    silly_module_name = TripWire('We do not have silly_module_name')
    with pytest.raises(TripWireError):
        silly_module_name.do_silly_thing
    try:
        silly_module_name.__wrapped__
    except TripWireError as err:
        assert isinstance(err, AttributeError)
    else:
        raise RuntimeError('No error raised, but expected')