import cirq
def test_virtual_tag():
    tag1 = cirq.ops.VirtualTag()
    tag2 = cirq.ops.VirtualTag()
    assert tag1 == tag2
    assert str(tag1) == str(tag2) == '<virtual>'
    cirq.testing.assert_equivalent_repr(tag1)
    cirq.testing.assert_equivalent_repr(tag2)