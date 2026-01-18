import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def test_write_network_text_empty_graph():

    def _graph_str(g, **kw):
        printbuf = []
        nx.write_network_text(g, printbuf.append, end='', **kw)
        return '\n'.join(printbuf)
    assert _graph_str(nx.DiGraph()) == '╙'
    assert _graph_str(nx.Graph()) == '╙'
    assert _graph_str(nx.DiGraph(), ascii_only=True) == '+'
    assert _graph_str(nx.Graph(), ascii_only=True) == '+'