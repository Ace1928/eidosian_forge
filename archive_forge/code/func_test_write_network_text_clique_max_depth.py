import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_clique_max_depth():
    orig = nx.complete_graph(5, nx.DiGraph)
    lines = []
    write = lines.append
    write('--- directed case, max_depth=None ---')
    nx.write_network_text(orig, path=write, end='', max_depth=None)
    write('--- directed case, max_depth=0 ---')
    nx.write_network_text(orig, path=write, end='', max_depth=0)
    write('--- directed case, max_depth=1 ---')
    nx.write_network_text(orig, path=write, end='', max_depth=1)
    write('--- directed case, max_depth=2 ---')
    nx.write_network_text(orig, path=write, end='', max_depth=2)
    write('--- directed case, max_depth=3 ---')
    nx.write_network_text(orig, path=write, end='', max_depth=3)
    write('--- undirected case, max_depth=None ---')
    nx.write_network_text(orig.to_undirected(), path=write, end='', max_depth=None)
    write('--- undirected case, max_depth=0 ---')
    nx.write_network_text(orig.to_undirected(), path=write, end='', max_depth=0)
    write('--- undirected case, max_depth=1 ---')
    nx.write_network_text(orig.to_undirected(), path=write, end='', max_depth=1)
    write('--- undirected case, max_depth=2 ---')
    nx.write_network_text(orig.to_undirected(), path=write, end='', max_depth=2)
    write('--- undirected case, max_depth=3 ---')
    nx.write_network_text(orig.to_undirected(), path=write, end='', max_depth=3)
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        --- directed case, max_depth=None ---\n        ╙── 0 ╾ 1, 2, 3, 4\n            ├─╼ 1 ╾ 2, 3, 4\n            │   ├─╼ 2 ╾ 0, 3, 4\n            │   │   ├─╼ 3 ╾ 0, 1, 4\n            │   │   │   ├─╼ 4 ╾ 0, 1, 2\n            │   │   │   │   └─╼  ...\n            │   │   │   └─╼  ...\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        --- directed case, max_depth=0 ---\n        ╙ ...\n        --- directed case, max_depth=1 ---\n        ╙── 0 ╾ 1, 2, 3, 4\n            └─╼  ...\n        --- directed case, max_depth=2 ---\n        ╙── 0 ╾ 1, 2, 3, 4\n            ├─╼ 1 ╾ 2, 3, 4\n            │   └─╼  ...\n            ├─╼ 2 ╾ 1, 3, 4\n            │   └─╼  ...\n            ├─╼ 3 ╾ 1, 2, 4\n            │   └─╼  ...\n            └─╼ 4 ╾ 1, 2, 3\n                └─╼  ...\n        --- directed case, max_depth=3 ---\n        ╙── 0 ╾ 1, 2, 3, 4\n            ├─╼ 1 ╾ 2, 3, 4\n            │   ├─╼ 2 ╾ 0, 3, 4\n            │   │   └─╼  ...\n            │   ├─╼ 3 ╾ 0, 2, 4\n            │   │   └─╼  ...\n            │   ├─╼ 4 ╾ 0, 2, 3\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        --- undirected case, max_depth=None ---\n        ╙── 0\n            ├── 1\n            │   ├── 2 ─ 0\n            │   │   ├── 3 ─ 0, 1\n            │   │   │   └── 4 ─ 0, 1, 2\n            │   │   └──  ...\n            │   └──  ...\n            └──  ...\n        --- undirected case, max_depth=0 ---\n        ╙ ...\n        --- undirected case, max_depth=1 ---\n        ╙── 0 ─ 1, 2, 3, 4\n            └──  ...\n        --- undirected case, max_depth=2 ---\n        ╙── 0\n            ├── 1 ─ 2, 3, 4\n            │   └──  ...\n            ├── 2 ─ 1, 3, 4\n            │   └──  ...\n            ├── 3 ─ 1, 2, 4\n            │   └──  ...\n            └── 4 ─ 1, 2, 3\n        --- undirected case, max_depth=3 ---\n        ╙── 0\n            ├── 1\n            │   ├── 2 ─ 0, 3, 4\n            │   │   └──  ...\n            │   ├── 3 ─ 0, 2, 4\n            │   │   └──  ...\n            │   └── 4 ─ 0, 2, 3\n            └──  ...\n        ').strip()
    assert target == text