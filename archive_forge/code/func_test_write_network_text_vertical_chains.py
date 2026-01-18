import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_vertical_chains():
    graph1 = nx.lollipop_graph(4, 2, create_using=nx.Graph)
    graph1.add_edge(0, -1)
    graph1.add_edge(-1, -2)
    graph1.add_edge(-2, -3)
    graph2 = graph1.to_directed()
    graph2.remove_edges_from([(u, v) for u, v in graph2.edges if v > u])
    lines = []
    write = lines.append
    write('--- Undirected UTF ---')
    nx.write_network_text(graph1, path=write, end='', vertical_chains=True)
    write('--- Undirected ASCI ---')
    nx.write_network_text(graph1, path=write, end='', vertical_chains=True, ascii_only=True)
    write('--- Directed UTF ---')
    nx.write_network_text(graph2, path=write, end='', vertical_chains=True)
    write('--- Directed ASCI ---')
    nx.write_network_text(graph2, path=write, end='', vertical_chains=True, ascii_only=True)
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        --- Undirected UTF ---\n        ╙── 5\n            │\n            4\n            │\n            3\n            ├── 0\n            │   ├── 1 ─ 3\n            │   │   │\n            │   │   2 ─ 0, 3\n            │   ├── -1\n            │   │   │\n            │   │   -2\n            │   │   │\n            │   │   -3\n            │   └──  ...\n            └──  ...\n        --- Undirected ASCI ---\n        +-- 5\n            |\n            4\n            |\n            3\n            |-- 0\n            |   |-- 1 - 3\n            |   |   |\n            |   |   2 - 0, 3\n            |   |-- -1\n            |   |   |\n            |   |   -2\n            |   |   |\n            |   |   -3\n            |   L--  ...\n            L--  ...\n        --- Directed UTF ---\n        ╙── 5\n            ╽\n            4\n            ╽\n            3\n            ├─╼ 0 ╾ 1, 2\n            │   ╽\n            │   -1\n            │   ╽\n            │   -2\n            │   ╽\n            │   -3\n            ├─╼ 1 ╾ 2\n            │   └─╼  ...\n            └─╼ 2\n                └─╼  ...\n        --- Directed ASCI ---\n        +-- 5\n            !\n            4\n            !\n            3\n            |-> 0 <- 1, 2\n            |   !\n            |   -1\n            |   !\n            |   -2\n            |   !\n            |   -3\n            |-> 1 <- 2\n            |   L->  ...\n            L-> 2\n                L->  ...\n        ').strip()
    assert target == text