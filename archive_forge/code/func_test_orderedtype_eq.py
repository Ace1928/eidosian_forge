from ..orderedtype import OrderedType
def test_orderedtype_eq():
    one = OrderedType()
    two = OrderedType()
    assert one == one
    assert one != two