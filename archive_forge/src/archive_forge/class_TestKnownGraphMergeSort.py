import pprint
from .. import _known_graph_py, errors, tests
from ..revision import NULL_REVISION
from . import features, test_graph
from .scenarios import load_tests_apply_scenarios
class TestKnownGraphMergeSort(TestCaseWithKnownGraph):

    def assertSortAndIterate(self, ancestry, branch_tip, result_list):
        """Check that merge based sorting and iter_topo_order on graph works."""
        graph = self.make_known_graph(ancestry)
        value = graph.merge_sort(branch_tip)
        value = [(n.key, n.merge_depth, n.revno, n.end_of_merge) for n in value]
        if result_list != value:
            self.assertEqualDiff(pprint.pformat(result_list), pprint.pformat(value))

    def test_merge_sort_empty(self):
        self.assertSortAndIterate({}, None, [])
        self.assertSortAndIterate({}, NULL_REVISION, [])
        self.assertSortAndIterate({}, (NULL_REVISION,), [])

    def test_merge_sort_not_empty_no_tip(self):
        self.assertSortAndIterate({0: []}, None, [])
        self.assertSortAndIterate({0: []}, NULL_REVISION, [])
        self.assertSortAndIterate({0: []}, (NULL_REVISION,), [])

    def test_merge_sort_one_revision(self):
        self.assertSortAndIterate({'id': []}, 'id', [('id', 0, (1,), True)])

    def test_sequence_numbers_increase_no_merges(self):
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['B']}, 'C', [('C', 0, (3,), False), ('B', 0, (2,), False), ('A', 0, (1,), True)])

    def test_sequence_numbers_increase_with_merges(self):
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['A', 'B']}, 'C', [('C', 0, (2,), False), ('B', 1, (1, 1, 1), True), ('A', 0, (1,), True)])

    def test_merge_sort_race(self):
        graph = {'A': [], 'B': ['A'], 'C': ['B'], 'D': ['B', 'C'], 'F': ['B', 'D']}
        self.assertSortAndIterate(graph, 'F', [('F', 0, (3,), False), ('D', 1, (2, 2, 1), False), ('C', 2, (2, 1, 1), True), ('B', 0, (2,), False), ('A', 0, (1,), True)])
        graph = {'A': [], 'B': ['A'], 'C': ['B'], 'X': ['B'], 'D': ['X', 'C'], 'F': ['B', 'D']}
        self.assertSortAndIterate(graph, 'F', [('F', 0, (3,), False), ('D', 1, (2, 1, 2), False), ('C', 2, (2, 2, 1), True), ('X', 1, (2, 1, 1), True), ('B', 0, (2,), False), ('A', 0, (1,), True)])

    def test_merge_depth_with_nested_merges(self):
        self.assertSortAndIterate({'A': ['D', 'B'], 'B': ['C', 'F'], 'C': ['H'], 'D': ['H', 'E'], 'E': ['G', 'F'], 'F': ['G'], 'G': ['H'], 'H': []}, 'A', [('A', 0, (3,), False), ('B', 1, (1, 3, 2), False), ('C', 1, (1, 3, 1), True), ('D', 0, (2,), False), ('E', 1, (1, 1, 2), False), ('F', 2, (1, 2, 1), True), ('G', 1, (1, 1, 1), True), ('H', 0, (1,), True)])

    def test_dotted_revnos_with_simple_merges(self):
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['A'], 'D': ['B'], 'E': ['C'], 'F': ['C'], 'G': ['D', 'E'], 'H': ['F'], 'I': ['F'], 'J': ['G', 'H'], 'K': ['I'], 'L': ['J', 'K']}, 'L', [('L', 0, (6,), False), ('K', 1, (1, 3, 2), False), ('I', 1, (1, 3, 1), True), ('J', 0, (5,), False), ('H', 1, (1, 2, 2), False), ('F', 1, (1, 2, 1), True), ('G', 0, (4,), False), ('E', 1, (1, 1, 2), False), ('C', 1, (1, 1, 1), True), ('D', 0, (3,), False), ('B', 0, (2,), False), ('A', 0, (1,), True)])
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['A'], 'D': ['B'], 'E': ['C'], 'F': ['C'], 'G': ['D', 'E'], 'H': ['F'], 'I': ['F'], 'J': ['G', 'H'], 'K': ['I'], 'L': ['J', 'K'], 'M': ['A'], 'N': ['L', 'M']}, 'N', [('N', 0, (7,), False), ('M', 1, (1, 4, 1), True), ('L', 0, (6,), False), ('K', 1, (1, 3, 2), False), ('I', 1, (1, 3, 1), True), ('J', 0, (5,), False), ('H', 1, (1, 2, 2), False), ('F', 1, (1, 2, 1), True), ('G', 0, (4,), False), ('E', 1, (1, 1, 2), False), ('C', 1, (1, 1, 1), True), ('D', 0, (3,), False), ('B', 0, (2,), False), ('A', 0, (1,), True)])

    def test_end_of_merge_not_last_revision_in_branch(self):
        self.assertSortAndIterate({'A': ['B'], 'B': []}, 'A', [('A', 0, (2,), False), ('B', 0, (1,), True)])

    def test_end_of_merge_multiple_revisions_merged_at_once(self):
        self.assertSortAndIterate({'A': ['H', 'B', 'E'], 'B': ['D', 'C'], 'C': ['D'], 'D': ['H'], 'E': ['G', 'F'], 'F': ['G'], 'G': ['H'], 'H': []}, 'A', [('A', 0, (2,), False), ('B', 1, (1, 3, 2), False), ('C', 2, (1, 4, 1), True), ('D', 1, (1, 3, 1), True), ('E', 1, (1, 1, 2), False), ('F', 2, (1, 2, 1), True), ('G', 1, (1, 1, 1), True), ('H', 0, (1,), True)])

    def test_parallel_root_sequence_numbers_increase_with_merges(self):
        """When there are parallel roots, check their revnos."""
        self.assertSortAndIterate({'A': [], 'B': [], 'C': ['A', 'B']}, 'C', [('C', 0, (2,), False), ('B', 1, (0, 1, 1), True), ('A', 0, (1,), True)])

    def test_revnos_are_globally_assigned(self):
        """revnos are assigned according to the revision they derive from."""
        self.assertSortAndIterate({'J': ['G', 'I'], 'I': ['H'], 'H': ['A'], 'G': ['D', 'F'], 'F': ['E'], 'E': ['A'], 'D': ['A', 'C'], 'C': ['B'], 'B': ['A'], 'A': []}, 'J', [('J', 0, (4,), False), ('I', 1, (1, 3, 2), False), ('H', 1, (1, 3, 1), True), ('G', 0, (3,), False), ('F', 1, (1, 2, 2), False), ('E', 1, (1, 2, 1), True), ('D', 0, (2,), False), ('C', 1, (1, 1, 2), False), ('B', 1, (1, 1, 1), True), ('A', 0, (1,), True)])

    def test_roots_and_sub_branches_versus_ghosts(self):
        """Extra roots and their mini branches use the same numbering.

        All of them use the 0-node numbering.
        """
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['B'], 'D': [], 'E': ['D'], 'F': ['D'], 'G': ['E', 'F'], 'H': ['C', 'G'], 'I': [], 'J': ['H', 'I'], 'K': [], 'L': ['K'], 'M': ['K'], 'N': ['L', 'M'], 'O': ['N'], 'P': ['N'], 'Q': ['O', 'P'], 'R': ['J', 'Q']}, 'R', [('R', 0, (6,), False), ('Q', 1, (0, 4, 5), False), ('P', 2, (0, 6, 1), True), ('O', 1, (0, 4, 4), False), ('N', 1, (0, 4, 3), False), ('M', 2, (0, 5, 1), True), ('L', 1, (0, 4, 2), False), ('K', 1, (0, 4, 1), True), ('J', 0, (5,), False), ('I', 1, (0, 3, 1), True), ('H', 0, (4,), False), ('G', 1, (0, 1, 3), False), ('F', 2, (0, 2, 1), True), ('E', 1, (0, 1, 2), False), ('D', 1, (0, 1, 1), True), ('C', 0, (3,), False), ('B', 0, (2,), False), ('A', 0, (1,), True)])

    def test_ghost(self):
        self.assertSortAndIterate({'A': [], 'B': ['A'], 'C': ['B', 'ghost']}, 'C', [('C', 0, (3,), False), ('B', 0, (2,), False), ('A', 0, (1,), True)])

    def test_lefthand_ghost(self):
        self.assertSortAndIterate({'A': ['ghost'], 'B': ['A']}, 'B', [('B', 0, (2,), False), ('A', 0, (1,), True)])

    def test_graph_cycle(self):
        self.assertRaises(errors.GraphCycleError, self.assertSortAndIterate, {'A': [], 'B': ['D'], 'C': ['B'], 'D': ['C'], 'E': ['D']}, 'E', [])