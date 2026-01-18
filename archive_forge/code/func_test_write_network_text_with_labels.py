import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_with_labels():
    graph = nx.generators.complete_graph(5, create_using=nx.DiGraph)
    for n in graph.nodes:
        graph.nodes[n]['label'] = f'Node(n={n})'
    lines = []
    write = lines.append
    nx.write_network_text(graph, path=write, with_labels=True, ascii_only=False, end='')
    text = '\n'.join(lines)
    print(text)
    target = dedent('\n        ╙── Node(n=0) ╾ Node(n=1), Node(n=2), Node(n=3), Node(n=4)\n            ├─╼ Node(n=1) ╾ Node(n=2), Node(n=3), Node(n=4)\n            │   ├─╼ Node(n=2) ╾ Node(n=0), Node(n=3), Node(n=4)\n            │   │   ├─╼ Node(n=3) ╾ Node(n=0), Node(n=1), Node(n=4)\n            │   │   │   ├─╼ Node(n=4) ╾ Node(n=0), Node(n=1), Node(n=2)\n            │   │   │   │   └─╼  ...\n            │   │   │   └─╼  ...\n            │   │   └─╼  ...\n            │   └─╼  ...\n            └─╼  ...\n        ').strip()
    assert target == text