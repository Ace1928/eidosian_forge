import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_within_forest_glyph():
    g = nx.DiGraph()
    g.add_nodes_from([1, 2, 3, 4])
    g.add_edge(2, 4)
    lines = []
    write = lines.append
    nx.write_network_text(g, path=write, end='')
    nx.write_network_text(g, path=write, ascii_only=True, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        ╟── 1\n        ╟── 2\n        ╎   └─╼ 4\n        ╙── 3\n        +-- 1\n        +-- 2\n        :   L-> 4\n        +-- 3\n        ').strip()
    assert text == target