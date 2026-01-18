import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_dorogovtsev_goltsev_mendes_graph():
    graph = nx.dorogovtsev_goltsev_mendes_graph(4, create_using=nx.Graph)
    lines = []
    write = lines.append
    nx.write_network_text(graph, path=write, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        ╙── 15\n            ├── 0\n            │   ├── 1 ─ 15\n            │   │   ├── 2 ─ 0\n            │   │   │   ├── 4 ─ 0\n            │   │   │   │   ├── 9 ─ 0\n            │   │   │   │   │   ├── 22 ─ 0\n            │   │   │   │   │   └── 38 ─ 4\n            │   │   │   │   ├── 13 ─ 2\n            │   │   │   │   │   ├── 34 ─ 2\n            │   │   │   │   │   └── 39 ─ 4\n            │   │   │   │   ├── 18 ─ 0\n            │   │   │   │   ├── 30 ─ 2\n            │   │   │   │   └──  ...\n            │   │   │   ├── 5 ─ 1\n            │   │   │   │   ├── 12 ─ 1\n            │   │   │   │   │   ├── 29 ─ 1\n            │   │   │   │   │   └── 40 ─ 5\n            │   │   │   │   ├── 14 ─ 2\n            │   │   │   │   │   ├── 35 ─ 2\n            │   │   │   │   │   └── 41 ─ 5\n            │   │   │   │   ├── 25 ─ 1\n            │   │   │   │   ├── 31 ─ 2\n            │   │   │   │   └──  ...\n            │   │   │   ├── 7 ─ 0\n            │   │   │   │   ├── 20 ─ 0\n            │   │   │   │   └── 32 ─ 2\n            │   │   │   ├── 10 ─ 1\n            │   │   │   │   ├── 27 ─ 1\n            │   │   │   │   └── 33 ─ 2\n            │   │   │   ├── 16 ─ 0\n            │   │   │   ├── 23 ─ 1\n            │   │   │   └──  ...\n            │   │   ├── 3 ─ 0\n            │   │   │   ├── 8 ─ 0\n            │   │   │   │   ├── 21 ─ 0\n            │   │   │   │   └── 36 ─ 3\n            │   │   │   ├── 11 ─ 1\n            │   │   │   │   ├── 28 ─ 1\n            │   │   │   │   └── 37 ─ 3\n            │   │   │   ├── 17 ─ 0\n            │   │   │   ├── 24 ─ 1\n            │   │   │   └──  ...\n            │   │   ├── 6 ─ 0\n            │   │   │   ├── 19 ─ 0\n            │   │   │   └── 26 ─ 1\n            │   │   └──  ...\n            │   └──  ...\n            └──  ...\n        ').strip()
    assert target == text