import pytest
import networkx as nx
def test_richclub2():
    T = nx.balanced_tree(2, 10)
    rc = nx.richclub.rich_club_coefficient(T, normalized=False)
    assert rc == {0: 4092 / (2047 * 2046.0), 1: 2044.0 / (1023 * 1022), 2: 2040.0 / (1022 * 1021)}