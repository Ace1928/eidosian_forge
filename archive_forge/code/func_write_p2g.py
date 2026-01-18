import networkx as nx
from networkx.utils import open_file
@open_file(1, mode='w')
def write_p2g(G, path, encoding='utf-8'):
    """Write NetworkX graph in p2g format.

    Notes
    -----
    This format is meant to be used with directed graphs with
    possible self loops.
    """
    path.write(f'{G.name}\n'.encode(encoding))
    path.write(f'{G.order()} {G.size()}\n'.encode(encoding))
    nodes = list(G)
    nodenumber = dict(zip(nodes, range(len(nodes))))
    for n in nodes:
        path.write(f'{n}\n'.encode(encoding))
        for nbr in G.neighbors(n):
            path.write(f'{nodenumber[nbr]} '.encode(encoding))
        path.write('\n'.encode(encoding))