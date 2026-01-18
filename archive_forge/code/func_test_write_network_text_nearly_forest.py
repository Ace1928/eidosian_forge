import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_nearly_forest():
    g = nx.DiGraph()
    g.add_edge(1, 2)
    g.add_edge(1, 5)
    g.add_edge(2, 3)
    g.add_edge(3, 4)
    g.add_edge(5, 6)
    g.add_edge(6, 7)
    g.add_edge(6, 8)
    orig = g.copy()
    g.add_edge(1, 8)
    g.add_edge(4, 2)
    g.add_edge(6, 3)
    lines = []
    write = lines.append
    write('--- directed case ---')
    nx.write_network_text(orig, path=write, end='')
    write('--- add (1, 8), (4, 2), (6, 3) ---')
    nx.write_network_text(g, path=write, end='')
    write('--- undirected case ---')
    nx.write_network_text(orig.to_undirected(), path=write, sources=[1], end='')
    write('--- add (1, 8), (4, 2), (6, 3) ---')
    nx.write_network_text(g.to_undirected(), path=write, sources=[1], end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        --- directed case ---\n        ╙── 1\n            ├─╼ 2\n            │   └─╼ 3\n            │       └─╼ 4\n            └─╼ 5\n                └─╼ 6\n                    ├─╼ 7\n                    └─╼ 8\n        --- add (1, 8), (4, 2), (6, 3) ---\n        ╙── 1\n            ├─╼ 2 ╾ 4\n            │   └─╼ 3 ╾ 6\n            │       └─╼ 4\n            │           └─╼  ...\n            ├─╼ 5\n            │   └─╼ 6\n            │       ├─╼ 7\n            │       ├─╼ 8 ╾ 1\n            │       └─╼  ...\n            └─╼  ...\n        --- undirected case ---\n        ╙── 1\n            ├── 2\n            │   └── 3\n            │       └── 4\n            └── 5\n                └── 6\n                    ├── 7\n                    └── 8\n        --- add (1, 8), (4, 2), (6, 3) ---\n        ╙── 1\n            ├── 2\n            │   ├── 3\n            │   │   ├── 4 ─ 2\n            │   │   └── 6\n            │   │       ├── 5 ─ 1\n            │   │       ├── 7\n            │   │       └── 8 ─ 1\n            │   └──  ...\n            └──  ...\n        ').strip()
    assert target == text