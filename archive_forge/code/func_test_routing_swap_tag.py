import cirq
def test_routing_swap_tag():
    tag1 = cirq.ops.RoutingSwapTag()
    tag2 = cirq.ops.RoutingSwapTag()
    assert tag1 == tag2
    assert str(tag1) == str(tag2) == '<r>'
    assert hash(tag1) == hash(tag2)
    cirq.testing.assert_equivalent_repr(tag1)
    cirq.testing.assert_equivalent_repr(tag2)