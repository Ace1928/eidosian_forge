from itertools import product
import pytest
import networkx as nx
def test_paper_2_case():
    ground_truth = {frozenset(['education', 'bs', 'ms', 'phd']), frozenset(['name', 'home_address']), frozenset(['telephone', 'home', 'office', 'no1', 'no2'])}
    tf = (True, False)
    for ewt, drc in product(tf, tf):
        part = paper_2_case(ewt, drc)
        assert part == ground_truth