import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_tree_max_depth():
    orig = nx.balanced_tree(r=1, h=3, create_using=nx.DiGraph)
    lines = []
    write = lines.append
    write('--- directed case, max_depth=0 ---')
    nx.write_network_text(orig, path=write, end='', max_depth=0)
    write('--- directed case, max_depth=1 ---')
    nx.write_network_text(orig, path=write, end='', max_depth=1)
    write('--- directed case, max_depth=2 ---')
    nx.write_network_text(orig, path=write, end='', max_depth=2)
    write('--- directed case, max_depth=3 ---')
    nx.write_network_text(orig, path=write, end='', max_depth=3)
    write('--- directed case, max_depth=4 ---')
    nx.write_network_text(orig, path=write, end='', max_depth=4)
    write('--- undirected case, max_depth=0 ---')
    nx.write_network_text(orig.to_undirected(), path=write, end='', max_depth=0)
    write('--- undirected case, max_depth=1 ---')
    nx.write_network_text(orig.to_undirected(), path=write, end='', max_depth=1)
    write('--- undirected case, max_depth=2 ---')
    nx.write_network_text(orig.to_undirected(), path=write, end='', max_depth=2)
    write('--- undirected case, max_depth=3 ---')
    nx.write_network_text(orig.to_undirected(), path=write, end='', max_depth=3)
    write('--- undirected case, max_depth=4 ---')
    nx.write_network_text(orig.to_undirected(), path=write, end='', max_depth=4)
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        --- directed case, max_depth=0 ---\n        ╙ ...\n        --- directed case, max_depth=1 ---\n        ╙── 0\n            └─╼  ...\n        --- directed case, max_depth=2 ---\n        ╙── 0\n            └─╼ 1\n                └─╼  ...\n        --- directed case, max_depth=3 ---\n        ╙── 0\n            └─╼ 1\n                └─╼ 2\n                    └─╼  ...\n        --- directed case, max_depth=4 ---\n        ╙── 0\n            └─╼ 1\n                └─╼ 2\n                    └─╼ 3\n        --- undirected case, max_depth=0 ---\n        ╙ ...\n        --- undirected case, max_depth=1 ---\n        ╙── 0 ─ 1\n            └──  ...\n        --- undirected case, max_depth=2 ---\n        ╙── 0\n            └── 1 ─ 2\n                └──  ...\n        --- undirected case, max_depth=3 ---\n        ╙── 0\n            └── 1\n                └── 2 ─ 3\n                    └──  ...\n        --- undirected case, max_depth=4 ---\n        ╙── 0\n            └── 1\n                └── 2\n                    └── 3\n        ').strip()
    assert target == text