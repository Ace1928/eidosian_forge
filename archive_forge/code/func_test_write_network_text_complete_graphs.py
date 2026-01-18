import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_complete_graphs():
    lines = []
    write = lines.append
    for k in [0, 1, 2, 3, 4, 5]:
        g = nx.generators.complete_graph(k)
        write(f'--- undirected k={k} ---')
        nx.write_network_text(g, path=write, end='')
    for k in [0, 1, 2, 3, 4, 5]:
        g = nx.generators.complete_graph(k, nx.DiGraph)
        write(f'--- directed k={k} ---')
        nx.write_network_text(g, path=write, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        --- undirected k=0 ---\n        ╙\n        --- undirected k=1 ---\n        ╙── 0\n        --- undirected k=2 ---\n        ╙── 0\n            └── 1\n        --- undirected k=3 ---\n        ╙── 0\n            ├── 1\n            │   └── 2 ─ 0\n            └──  ...\n        --- undirected k=4 ---\n        ╙── 0\n            ├── 1\n            │   ├── 2 ─ 0\n            │   │   └── 3 ─ 0, 1\n            │   └──  ...\n            └──  ...\n        --- undirected k=5 ---\n        ╙── 0\n            ├── 1\n            │   ├── 2 ─ 0\n            │   │   ├── 3 ─ 0, 1\n            │   │   │   └── 4 ─ 0, 1, 2\n            │   │   └──  ...\n            │   └──  ...\n            └──  ...\n        --- directed k=0 ---\n        ╙\n        --- directed k=1 ---\n        ╙── 0\n        --- directed k=2 ---\n        ╙── 0 ╾ 1\n            └─╼ 1\n                └─╼  ...\n        --- directed k=3 ---\n        ╙── 0 ╾ 1, 2\n            ├─╼ 1 ╾ 2\n            │   ├─╼ 2 ╾ 0\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        --- directed k=4 ---\n        ╙── 0 ╾ 1, 2, 3\n            ├─╼ 1 ╾ 2, 3\n            │   ├─╼ 2 ╾ 0, 3\n            │   │   ├─╼ 3 ╾ 0, 1\n            │   │   │   └─╼  ...\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        --- directed k=5 ---\n        ╙── 0 ╾ 1, 2, 3, 4\n            ├─╼ 1 ╾ 2, 3, 4\n            │   ├─╼ 2 ╾ 0, 3, 4\n            │   │   ├─╼ 3 ╾ 0, 1, 4\n            │   │   │   ├─╼ 4 ╾ 0, 1, 2\n            │   │   │   │   └─╼  ...\n            │   │   │   └─╼  ...\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        ').strip()
    assert target == text