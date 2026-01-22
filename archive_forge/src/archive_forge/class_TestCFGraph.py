import itertools
import unittest
from numba import jit
from numba.core.controlflow import CFGraph, ControlFlowAnalysis
from numba.core import types
from numba.core.bytecode import FunctionIdentity, ByteCode, _fix_LOAD_GLOBAL_arg
from numba.core.utils import PYVERSION
from numba.tests.support import TestCase
class TestCFGraph(TestCase):
    """
    Test the numba.controlflow.CFGraph class.
    """

    def from_adj_list(self, d, entry_point=0):
        """
        Build a CFGraph class from a dict of adjacency lists.
        """
        g = CFGraph()
        for node in d:
            g.add_node(node)
        for node, dests in d.items():
            for dest in dests:
                g.add_edge(node, dest)
        return g

    def loopless1(self):
        """
        A simple CFG corresponding to the following code structure:

            c = (... if ... else ...) + ...
            return b + c
        """
        g = self.from_adj_list({0: [18, 12], 12: [21], 18: [21], 21: []})
        g.set_entry_point(0)
        g.process()
        return g

    def loopless1_dead_nodes(self):
        """
        Same as loopless1(), but with added dead blocks (some of them
        in a loop).
        """
        g = self.from_adj_list({0: [18, 12], 12: [21], 18: [21], 21: [], 91: [12, 0], 92: [91, 93], 93: [92], 94: []})
        g.set_entry_point(0)
        g.process()
        return g

    def loopless2(self):
        """
        A loopless CFG corresponding to the following code structure:

            c = (... if ... else ...) + ...
            if c:
                return ...
            else:
                return ...

        Note there are two exit points, and the entry point has been
        changed to a non-zero value.
        """
        g = self.from_adj_list({99: [18, 12], 12: [21], 18: [21], 21: [42, 34], 34: [], 42: []})
        g.set_entry_point(99)
        g.process()
        return g

    def multiple_loops(self):
        """
        A CFG with multiple nested loops:

            for y in b:
                for x in a:
                    # This loop has two back edges
                    if b:
                        continue
                    else:
                        continue
            for z in c:
                if z:
                    return ...
        """
        g = self.from_adj_list({0: [7], 7: [10, 60], 10: [13], 13: [20], 20: [56, 23], 23: [32, 44], 32: [20], 44: [20], 56: [57], 57: [7], 60: [61], 61: [68], 68: [87, 71], 71: [80, 68], 80: [], 87: [88], 88: []})
        g.set_entry_point(0)
        g.process()
        return g

    def multiple_exits(self):
        """
        A CFG with three loop exits, one of which is also a function
        exit point, and another function exit point:

            for x in a:
                if a:
                    return b
                elif b:
                    break
            return c
        """
        g = self.from_adj_list({0: [7], 7: [10, 36], 10: [19, 23], 19: [], 23: [29, 7], 29: [37], 36: [37], 37: []})
        g.set_entry_point(0)
        g.process()
        return g

    def infinite_loop1(self):
        """
        A CFG with a infinite loop and an alternate exit point:

            if c:
                return
            while True:
                if a:
                    ...
                else:
                    ...
        """
        g = self.from_adj_list({0: [10, 6], 6: [], 10: [13], 13: [26, 19], 19: [13], 26: [13]})
        g.set_entry_point(0)
        g.process()
        return g

    def infinite_loop2(self):
        """
        A CFG with no exit point at all:

            while True:
                if a:
                    ...
                else:
                    ...
        """
        g = self.from_adj_list({0: [3], 3: [16, 9], 9: [3], 16: [3]})
        g.set_entry_point(0)
        g.process()
        return g

    def test_simple_properties(self):
        g = self.loopless1()
        self.assertEqual(sorted(g.successors(0)), [(12, None), (18, None)])
        self.assertEqual(sorted(g.successors(21)), [])
        self.assertEqual(sorted(g.predecessors(0)), [])
        self.assertEqual(sorted(g.predecessors(21)), [(12, None), (18, None)])

    def test_exit_points(self):
        g = self.loopless1()
        self.assertEqual(sorted(g.exit_points()), [21])
        g = self.loopless1_dead_nodes()
        self.assertEqual(sorted(g.exit_points()), [21])
        g = self.loopless2()
        self.assertEqual(sorted(g.exit_points()), [34, 42])
        g = self.multiple_loops()
        self.assertEqual(sorted(g.exit_points()), [80, 88])
        g = self.infinite_loop1()
        self.assertEqual(sorted(g.exit_points()), [6])
        g = self.infinite_loop2()
        self.assertEqual(sorted(g.exit_points()), [])
        g = self.multiple_exits()
        self.assertEqual(sorted(g.exit_points()), [19, 37])

    def test_dead_nodes(self):
        g = self.loopless1()
        self.assertEqual(len(g.dead_nodes()), 0)
        self.assertEqual(sorted(g.nodes()), [0, 12, 18, 21])
        g = self.loopless2()
        self.assertEqual(len(g.dead_nodes()), 0)
        self.assertEqual(sorted(g.nodes()), [12, 18, 21, 34, 42, 99])
        g = self.multiple_loops()
        self.assertEqual(len(g.dead_nodes()), 0)
        g = self.infinite_loop1()
        self.assertEqual(len(g.dead_nodes()), 0)
        g = self.multiple_exits()
        self.assertEqual(len(g.dead_nodes()), 0)
        g = self.loopless1_dead_nodes()
        self.assertEqual(sorted(g.dead_nodes()), [91, 92, 93, 94])
        self.assertEqual(sorted(g.nodes()), [0, 12, 18, 21])

    def test_descendents(self):
        g = self.loopless2()
        d = g.descendents(34)
        self.assertEqual(sorted(d), [])
        d = g.descendents(42)
        self.assertEqual(sorted(d), [])
        d = g.descendents(21)
        self.assertEqual(sorted(d), [34, 42])
        d = g.descendents(99)
        self.assertEqual(sorted(d), [12, 18, 21, 34, 42])
        g = self.infinite_loop1()
        d = g.descendents(26)
        self.assertEqual(sorted(d), [])
        d = g.descendents(19)
        self.assertEqual(sorted(d), [])
        d = g.descendents(13)
        self.assertEqual(sorted(d), [19, 26])
        d = g.descendents(10)
        self.assertEqual(sorted(d), [13, 19, 26])
        d = g.descendents(6)
        self.assertEqual(sorted(d), [])
        d = g.descendents(0)
        self.assertEqual(sorted(d), [6, 10, 13, 19, 26])

    def test_topo_order(self):
        g = self.loopless1()
        self.assertIn(g.topo_order(), ([0, 12, 18, 21], [0, 18, 12, 21]))
        g = self.loopless2()
        self.assertIn(g.topo_order(), ([99, 18, 12, 21, 34, 42], [99, 12, 18, 21, 34, 42]))
        g = self.infinite_loop2()
        self.assertIn(g.topo_order(), ([0, 3, 9, 16], [0, 3, 16, 9]))
        g = self.infinite_loop1()
        self.assertIn(g.topo_order(), ([0, 6, 10, 13, 19, 26], [0, 6, 10, 13, 26, 19], [0, 10, 13, 19, 26, 6], [0, 10, 13, 26, 19, 6]))

    def test_topo_sort(self):

        def check_topo_sort(nodes, expected):
            self.assertIn(list(g.topo_sort(nodes)), expected)
            self.assertIn(list(g.topo_sort(nodes[::-1])), expected)
            self.assertIn(list(g.topo_sort(nodes, reverse=True))[::-1], expected)
            self.assertIn(list(g.topo_sort(nodes[::-1], reverse=True))[::-1], expected)
            self.random.shuffle(nodes)
            self.assertIn(list(g.topo_sort(nodes)), expected)
            self.assertIn(list(g.topo_sort(nodes, reverse=True))[::-1], expected)
        g = self.loopless2()
        check_topo_sort([21, 99, 12, 34], ([99, 12, 21, 34],))
        check_topo_sort([18, 12, 42, 99], ([99, 12, 18, 42], [99, 18, 12, 42]))
        g = self.multiple_exits()
        check_topo_sort([19, 10, 7, 36], ([7, 10, 19, 36], [7, 10, 36, 19], [7, 36, 10, 19]))

    def check_dominators(self, got, expected):
        self.assertEqual(sorted(got), sorted(expected))
        for node in sorted(got):
            self.assertEqual(sorted(got[node]), sorted(expected[node]), 'mismatch for %r' % (node,))

    def test_dominators_loopless(self):

        def eq_(d, l):
            self.assertEqual(sorted(doms[d]), l)
        for g in [self.loopless1(), self.loopless1_dead_nodes()]:
            doms = g.dominators()
            eq_(0, [0])
            eq_(12, [0, 12])
            eq_(18, [0, 18])
            eq_(21, [0, 21])
        g = self.loopless2()
        doms = g.dominators()
        eq_(99, [99])
        eq_(12, [12, 99])
        eq_(18, [18, 99])
        eq_(21, [21, 99])
        eq_(34, [21, 34, 99])
        eq_(42, [21, 42, 99])

    def test_dominators_loops(self):
        g = self.multiple_exits()
        doms = g.dominators()
        self.check_dominators(doms, {0: [0], 7: [0, 7], 10: [0, 7, 10], 19: [0, 7, 10, 19], 23: [0, 7, 10, 23], 29: [0, 7, 10, 23, 29], 36: [0, 7, 36], 37: [0, 7, 37]})
        g = self.multiple_loops()
        doms = g.dominators()
        self.check_dominators(doms, {0: [0], 7: [0, 7], 10: [0, 10, 7], 13: [0, 10, 13, 7], 20: [0, 10, 20, 13, 7], 23: [0, 20, 23, 7, 10, 13], 32: [32, 0, 20, 23, 7, 10, 13], 44: [0, 20, 23, 7, 10, 44, 13], 56: [0, 20, 7, 56, 10, 13], 57: [0, 20, 7, 56, 57, 10, 13], 60: [0, 60, 7], 61: [0, 60, 61, 7], 68: [0, 68, 60, 61, 7], 71: [0, 68, 71, 7, 60, 61], 80: [80, 0, 68, 71, 7, 60, 61], 87: [0, 68, 87, 7, 60, 61], 88: [0, 68, 87, 88, 7, 60, 61]})
        g = self.infinite_loop1()
        doms = g.dominators()
        self.check_dominators(doms, {0: [0], 6: [0, 6], 10: [0, 10], 13: [0, 10, 13], 19: [0, 10, 19, 13], 26: [0, 10, 13, 26]})

    def test_post_dominators_loopless(self):

        def eq_(d, l):
            self.assertEqual(sorted(doms[d]), l)
        for g in [self.loopless1(), self.loopless1_dead_nodes()]:
            doms = g.post_dominators()
            eq_(0, [0, 21])
            eq_(12, [12, 21])
            eq_(18, [18, 21])
            eq_(21, [21])
        g = self.loopless2()
        doms = g.post_dominators()
        eq_(34, [34])
        eq_(42, [42])
        eq_(21, [21])
        eq_(18, [18, 21])
        eq_(12, [12, 21])
        eq_(99, [21, 99])

    def test_post_dominators_loops(self):
        g = self.multiple_exits()
        doms = g.post_dominators()
        self.check_dominators(doms, {0: [0, 7], 7: [7], 10: [10], 19: [19], 23: [23], 29: [29, 37], 36: [36, 37], 37: [37]})
        g = self.multiple_loops()
        doms = g.post_dominators()
        self.check_dominators(doms, {0: [0, 60, 68, 61, 7], 7: [60, 68, 61, 7], 10: [68, 7, 10, 13, 20, 56, 57, 60, 61], 13: [68, 7, 13, 20, 56, 57, 60, 61], 20: [20, 68, 7, 56, 57, 60, 61], 23: [68, 7, 20, 23, 56, 57, 60, 61], 32: [32, 68, 7, 20, 56, 57, 60, 61], 44: [68, 7, 44, 20, 56, 57, 60, 61], 56: [68, 7, 56, 57, 60, 61], 57: [57, 60, 68, 61, 7], 60: [60, 68, 61], 61: [68, 61], 68: [68], 71: [71], 80: [80], 87: [88, 87], 88: [88]})

    def test_post_dominators_infinite_loops(self):
        g = self.infinite_loop1()
        doms = g.post_dominators()
        self.check_dominators(doms, {0: [0], 6: [6], 10: [10, 13], 13: [13], 19: [19], 26: [26]})
        g = self.infinite_loop2()
        doms = g.post_dominators()
        self.check_dominators(doms, {0: [0, 3], 3: [3], 9: [9], 16: [16]})

    def test_dominator_tree(self):

        def check(graph, expected):
            domtree = graph.dominator_tree()
            self.assertEqual(domtree, expected)
        check(self.loopless1(), {0: {12, 18, 21}, 12: set(), 18: set(), 21: set()})
        check(self.loopless2(), {12: set(), 18: set(), 21: {34, 42}, 34: set(), 42: set(), 99: {18, 12, 21}})
        check(self.loopless1_dead_nodes(), {0: {12, 18, 21}, 12: set(), 18: set(), 21: set()})
        check(self.multiple_loops(), {0: {7}, 7: {10, 60}, 60: {61}, 61: {68}, 68: {71, 87}, 87: {88}, 88: set(), 71: {80}, 80: set(), 10: {13}, 13: {20}, 20: {56, 23}, 23: {32, 44}, 44: set(), 32: set(), 56: {57}, 57: set()})
        check(self.multiple_exits(), {0: {7}, 7: {10, 36, 37}, 36: set(), 10: {19, 23}, 23: {29}, 29: set(), 37: set(), 19: set()})
        check(self.infinite_loop1(), {0: {10, 6}, 6: set(), 10: {13}, 13: {26, 19}, 19: set(), 26: set()})
        check(self.infinite_loop2(), {0: {3}, 3: {16, 9}, 9: set(), 16: set()})

    def test_immediate_dominators(self):

        def check(graph, expected):
            idoms = graph.immediate_dominators()
            self.assertEqual(idoms, expected)
        check(self.loopless1(), {0: 0, 12: 0, 18: 0, 21: 0})
        check(self.loopless2(), {18: 99, 12: 99, 21: 99, 42: 21, 34: 21, 99: 99})
        check(self.loopless1_dead_nodes(), {0: 0, 12: 0, 18: 0, 21: 0})
        check(self.multiple_loops(), {0: 0, 7: 0, 10: 7, 13: 10, 20: 13, 23: 20, 32: 23, 44: 23, 56: 20, 57: 56, 60: 7, 61: 60, 68: 61, 71: 68, 80: 71, 87: 68, 88: 87})
        check(self.multiple_exits(), {0: 0, 7: 0, 10: 7, 19: 10, 23: 10, 29: 23, 36: 7, 37: 7})
        check(self.infinite_loop1(), {0: 0, 6: 0, 10: 0, 13: 10, 19: 13, 26: 13})
        check(self.infinite_loop2(), {0: 0, 3: 0, 9: 3, 16: 3})

    def test_dominance_frontier(self):

        def check(graph, expected):
            df = graph.dominance_frontier()
            self.assertEqual(df, expected)
        check(self.loopless1(), {0: set(), 12: {21}, 18: {21}, 21: set()})
        check(self.loopless2(), {18: {21}, 12: {21}, 21: set(), 42: set(), 34: set(), 99: set()})
        check(self.loopless1_dead_nodes(), {0: set(), 12: {21}, 18: {21}, 21: set()})
        check(self.multiple_loops(), {0: set(), 7: {7}, 10: {7}, 13: {7}, 20: {20, 7}, 23: {20}, 32: {20}, 44: {20}, 56: {7}, 57: {7}, 60: set(), 61: set(), 68: {68}, 71: {68}, 80: set(), 87: set(), 88: set()})
        check(self.multiple_exits(), {0: set(), 7: {7}, 10: {37, 7}, 19: set(), 23: {37, 7}, 29: {37}, 36: {37}, 37: set()})
        check(self.infinite_loop1(), {0: set(), 6: set(), 10: set(), 13: {13}, 19: {13}, 26: {13}})
        check(self.infinite_loop2(), {0: set(), 3: {3}, 9: {3}, 16: {3}})

    def test_backbone_loopless(self):
        for g in [self.loopless1(), self.loopless1_dead_nodes()]:
            self.assertEqual(sorted(g.backbone()), [0, 21])
        g = self.loopless2()
        self.assertEqual(sorted(g.backbone()), [21, 99])

    def test_backbone_loops(self):
        g = self.multiple_loops()
        self.assertEqual(sorted(g.backbone()), [0, 7, 60, 61, 68])
        g = self.infinite_loop1()
        self.assertEqual(sorted(g.backbone()), [0])
        g = self.infinite_loop2()
        self.assertEqual(sorted(g.backbone()), [0, 3])

    def test_loops(self):
        for g in [self.loopless1(), self.loopless1_dead_nodes(), self.loopless2()]:
            self.assertEqual(len(g.loops()), 0)
        g = self.multiple_loops()
        self.assertEqual(sorted(g.loops()), [7, 20, 68])
        outer1 = g.loops()[7]
        inner1 = g.loops()[20]
        outer2 = g.loops()[68]
        self.assertEqual(outer1.header, 7)
        self.assertEqual(sorted(outer1.entries), [0])
        self.assertEqual(sorted(outer1.exits), [60])
        self.assertEqual(sorted(outer1.body), [7, 10, 13, 20, 23, 32, 44, 56, 57])
        self.assertEqual(inner1.header, 20)
        self.assertEqual(sorted(inner1.entries), [13])
        self.assertEqual(sorted(inner1.exits), [56])
        self.assertEqual(sorted(inner1.body), [20, 23, 32, 44])
        self.assertEqual(outer2.header, 68)
        self.assertEqual(sorted(outer2.entries), [61])
        self.assertEqual(sorted(outer2.exits), [80, 87])
        self.assertEqual(sorted(outer2.body), [68, 71])
        for node in [0, 60, 61, 80, 87, 88]:
            self.assertEqual(g.in_loops(node), [])
        for node in [7, 10, 13, 56, 57]:
            self.assertEqual(g.in_loops(node), [outer1])
        for node in [20, 23, 32, 44]:
            self.assertEqual(g.in_loops(node), [inner1, outer1])
        for node in [68, 71]:
            self.assertEqual(g.in_loops(node), [outer2])
        g = self.infinite_loop1()
        self.assertEqual(sorted(g.loops()), [13])
        loop = g.loops()[13]
        self.assertEqual(loop.header, 13)
        self.assertEqual(sorted(loop.entries), [10])
        self.assertEqual(sorted(loop.exits), [])
        self.assertEqual(sorted(loop.body), [13, 19, 26])
        for node in [0, 6, 10]:
            self.assertEqual(g.in_loops(node), [])
        for node in [13, 19, 26]:
            self.assertEqual(g.in_loops(node), [loop])
        g = self.infinite_loop2()
        self.assertEqual(sorted(g.loops()), [3])
        loop = g.loops()[3]
        self.assertEqual(loop.header, 3)
        self.assertEqual(sorted(loop.entries), [0])
        self.assertEqual(sorted(loop.exits), [])
        self.assertEqual(sorted(loop.body), [3, 9, 16])
        for node in [0]:
            self.assertEqual(g.in_loops(node), [])
        for node in [3, 9, 16]:
            self.assertEqual(g.in_loops(node), [loop])
        g = self.multiple_exits()
        self.assertEqual(sorted(g.loops()), [7])
        loop = g.loops()[7]
        self.assertEqual(loop.header, 7)
        self.assertEqual(sorted(loop.entries), [0])
        self.assertEqual(sorted(loop.exits), [19, 29, 36])
        self.assertEqual(sorted(loop.body), [7, 10, 23])
        for node in [0, 19, 29, 36]:
            self.assertEqual(g.in_loops(node), [])
        for node in [7, 10, 23]:
            self.assertEqual(g.in_loops(node), [loop])

    def test_loop_dfs_pathological(self):
        g = self.from_adj_list({0: {38, 14}, 14: {38, 22}, 22: {38, 30}, 30: {42, 38}, 38: {42}, 42: {64, 50}, 50: {64, 58}, 58: {128}, 64: {72, 86}, 72: {80, 86}, 80: {128}, 86: {108, 94}, 94: {108, 102}, 102: {128}, 108: {128, 116}, 116: {128, 124}, 124: {128}, 128: {178, 174}, 174: {178}, 178: {210, 206}, 206: {210}, 210: {248, 252}, 248: {252}, 252: {282, 286}, 282: {286}, 286: {296, 326}, 296: {330}, 326: {330}, 330: {370, 340}, 340: {374}, 370: {374}, 374: {380, 382}, 380: {382}, 382: {818, 390}, 390: {456, 458}, 456: {458}, 458: {538, 566}, 538: {548, 566}, 548: set(), 566: {586, 572}, 572: {586}, 586: {708, 596}, 596: {608}, 608: {610}, 610: {704, 620}, 620: {666, 630}, 630: {636, 646}, 636: {666, 646}, 646: {666}, 666: {610}, 704: {706}, 706: {818}, 708: {720}, 720: {722}, 722: {816, 732}, 732: {778, 742}, 742: {748, 758}, 748: {778, 758}, 758: {778}, 778: {722}, 816: {818}, 818: set()})
        g.set_entry_point(0)
        g.process()
        stats = {}
        back_edges = g._find_back_edges(stats=stats)
        self.assertEqual(back_edges, {(666, 610), (778, 722)})
        self.assertEqual(stats['iteration_count'], 155)

    def test_equals(self):

        def get_new():
            g = self.from_adj_list({0: [18, 12], 12: [21], 18: [21], 21: []})
            g.set_entry_point(0)
            g.process()
            return g
        x = get_new()
        y = get_new()
        self.assertEqual(x, y)
        g = self.from_adj_list({0: [12, 18], 18: [21], 21: [], 12: [21]})
        g.set_entry_point(0)
        g.process()
        self.assertEqual(x, g)
        z = get_new()
        z.set_entry_point(18)
        z.process()
        self.assertNotEqual(x, z)
        z = self.from_adj_list({0: [18, 12], 12: [21], 18: [21], 21: [22], 22: []})
        z.set_entry_point(0)
        z.process()
        self.assertNotEqual(x, z)
        a = self.from_adj_list({0: [18, 12], 12: [0], 18: []})
        a.set_entry_point(0)
        a.process()
        z = self.from_adj_list({0: [18, 12], 12: [18], 18: []})
        z.set_entry_point(0)
        z.process()
        self.assertNotEqual(a, z)