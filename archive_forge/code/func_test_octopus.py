from dulwich.tests import TestCase
from ..graph import WorkList, _find_lcas, can_fast_forward
from ..repo import MemoryRepo
from .utils import make_commit
def test_octopus(self):
    graph = {'C': ['C1'], 'C1': ['C2'], 'C2': ['C3'], 'C3': ['C4'], 'C4': ['2'], 'B': ['B1'], 'B1': ['B2'], 'B2': ['B3'], 'B3': ['1'], 'A': ['A1'], 'A1': ['A2'], 'A2': ['A3'], 'A3': ['1'], '1': ['2'], '2': []}

    def lookup_parents(cid):
        return graph[cid]

    def lookup_stamp(commit_id):
        return 100
    lcas = ['A']
    others = ['B', 'C']
    for cmt in others:
        next_lcas = []
        for ca in lcas:
            res = _find_lcas(lookup_parents, cmt, [ca], lookup_stamp)
            next_lcas.extend(res)
        lcas = next_lcas[:]
    self.assertEqual(set(lcas), {'2'})