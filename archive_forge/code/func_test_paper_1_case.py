from itertools import product
import pytest
import networkx as nx
def test_paper_1_case():
    ground_truth = {frozenset([1, 4]), frozenset([2, 3, 5])}
    tf = (True, False)
    for flt, nwt, drc in product(tf, tf, tf):
        part = paper_1_case(flt, nwt, drc)
        assert part == ground_truth