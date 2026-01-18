import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_complete_graph_ascii_only():
    graph = nx.generators.complete_graph(5, create_using=nx.DiGraph)
    lines = []
    write = lines.append
    write('--- directed case ---')
    nx.write_network_text(graph, path=write, ascii_only=True, end='')
    write('--- undirected case ---')
    nx.write_network_text(graph.to_undirected(), path=write, ascii_only=True, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        --- directed case ---\n        +-- 0 <- 1, 2, 3, 4\n            |-> 1 <- 2, 3, 4\n            |   |-> 2 <- 0, 3, 4\n            |   |   |-> 3 <- 0, 1, 4\n            |   |   |   |-> 4 <- 0, 1, 2\n            |   |   |   |   L->  ...\n            |   |   |   L->  ...\n            |   |   L->  ...\n            |   L->  ...\n            L->  ...\n        --- undirected case ---\n        +-- 0\n            |-- 1\n            |   |-- 2 - 0\n            |   |   |-- 3 - 0, 1\n            |   |   |   L-- 4 - 0, 1, 2\n            |   |   L--  ...\n            |   L--  ...\n            L--  ...\n        ').strip()
    assert target == text