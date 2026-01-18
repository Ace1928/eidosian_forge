import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_custom_label():
    graph = nx.erdos_renyi_graph(5, 0.4, directed=True, seed=359222358)
    for node in graph.nodes:
        graph.nodes[node]['label'] = f'Node({node})'
        graph.nodes[node]['chr'] = chr(node + ord('a') - 1)
        if node % 2 == 0:
            graph.nodes[node]['part'] = chr(node + ord('a'))
    lines = []
    write = lines.append
    write("--- when with_labels=True, uses the 'label' attr ---")
    nx.write_network_text(graph, path=write, with_labels=True, end='', max_depth=None)
    write('--- when with_labels=False, uses str(node) value ---')
    nx.write_network_text(graph, path=write, with_labels=False, end='', max_depth=None)
    write('--- when with_labels is a string, use that attr ---')
    nx.write_network_text(graph, path=write, with_labels='chr', end='', max_depth=None)
    write('--- fallback to str(node) when the attr does not exist ---')
    nx.write_network_text(graph, path=write, with_labels='part', end='', max_depth=None)
    text = '\n'.join(lines)
    print(text)
    target = dedent("\n        --- when with_labels=True, uses the 'label' attr ---\n        ╙── Node(1)\n            └─╼ Node(3) ╾ Node(2)\n                ├─╼ Node(0)\n                │   ├─╼ Node(2) ╾ Node(3), Node(4)\n                │   │   └─╼  ...\n                │   └─╼ Node(4)\n                │       └─╼  ...\n                └─╼  ...\n        --- when with_labels=False, uses str(node) value ---\n        ╙── 1\n            └─╼ 3 ╾ 2\n                ├─╼ 0\n                │   ├─╼ 2 ╾ 3, 4\n                │   │   └─╼  ...\n                │   └─╼ 4\n                │       └─╼  ...\n                └─╼  ...\n        --- when with_labels is a string, use that attr ---\n        ╙── a\n            └─╼ c ╾ b\n                ├─╼ `\n                │   ├─╼ b ╾ c, d\n                │   │   └─╼  ...\n                │   └─╼ d\n                │       └─╼  ...\n                └─╼  ...\n        --- fallback to str(node) when the attr does not exist ---\n        ╙── 1\n            └─╼ 3 ╾ c\n                ├─╼ a\n                │   ├─╼ c ╾ 3, e\n                │   │   └─╼  ...\n                │   └─╼ e\n                │       └─╼  ...\n                └─╼  ...\n        ").strip()
    assert target == text