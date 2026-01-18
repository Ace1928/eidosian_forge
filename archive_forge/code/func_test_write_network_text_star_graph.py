import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_star_graph():
    graph = nx.star_graph(5, create_using=nx.Graph)
    lines = []
    write = lines.append
    nx.write_network_text(graph, path=write, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        ╙── 1\n            └── 0\n                ├── 2\n                ├── 3\n                ├── 4\n                └── 5\n        ').strip()
    assert target == text