import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_remove_redundant_acquaintance_opportunities():
    a, b, c, d, e = cirq.LineQubit.range(5)
    swap = cca.SwapPermutationGate()
    ops = [cca.acquaint(a, b), cca.acquaint(a, b)]
    strategy = cirq.Circuit(ops)
    diagram_before = '\n0: ───█───█───\n      │   │\n1: ───█───█───\n    '
    ct.assert_has_diagram(strategy, diagram_before)
    cca.remove_redundant_acquaintance_opportunities(strategy)
    diagram_after = '\n0: ───█───────\n      │\n1: ───█───────\n    '
    ct.assert_has_diagram(strategy, diagram_after)
    ops = [cca.acquaint(a, b), cca.acquaint(c, d), swap(d, e), swap(c, d), cca.acquaint(d, e)]
    strategy = cirq.Circuit(ops)
    diagram_before = '\n0: ───█───────────────────\n      │\n1: ───█───────────────────\n\n2: ───█─────────0↦1───────\n      │         │\n3: ───█───0↦1───1↦0───█───\n          │           │\n4: ───────1↦0─────────█───\n    '
    ct.assert_has_diagram(strategy, diagram_before)
    cca.remove_redundant_acquaintance_opportunities(strategy)
    diagram_after = '\n0: ───█───────────────────\n      │\n1: ───█───────────────────\n\n2: ───█─────────0↦1───────\n      │         │\n3: ───█───0↦1───1↦0───────\n          │\n4: ───────1↦0─────────────\n    '
    ct.assert_has_diagram(strategy, diagram_after)