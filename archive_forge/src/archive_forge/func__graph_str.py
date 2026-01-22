import random
from itertools import product
from textwrap import dedent
import pytest
import networkx as nx
def _graph_str(g, **kw):
    printbuf = []
    nx.write_network_text(g, printbuf.append, end='', **kw)
    return '\n'.join(printbuf)