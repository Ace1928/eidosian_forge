from ..orderedtype import OrderedType
def test_orderedtype_resetcounter():
    one = OrderedType()
    two = OrderedType()
    one.reset_counter()
    assert one > two