from ..orderedtype import OrderedType
def test_orderedtype_non_orderabletypes():
    one = OrderedType()
    assert one.__lt__(1) == NotImplemented
    assert one.__gt__(1) == NotImplemented
    assert one != 1