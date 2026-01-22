import gzip
import importlib.resources
import os
import os.path
from itertools import islice
import networkx as nx
def _generate_graphs():
    """Sequentially read the file containing the edge list data for the
    graphs in the atlas and generate the graphs one at a time.

    This function reads the file given in :data:`.ATLAS_FILE`.

    """
    with gzip.open(ATLAS_FILE, 'rb') as f:
        line = f.readline()
        while line and line.startswith(b'GRAPH'):
            graph_index = int(line[6:].rstrip())
            line = f.readline()
            num_nodes = int(line[6:].rstrip())
            edgelist = []
            line = f.readline()
            while line and (not line.startswith(b'GRAPH')):
                edgelist.append(line.rstrip())
                line = f.readline()
            G = nx.Graph()
            G.name = f'G{graph_index}'
            G.add_nodes_from(range(num_nodes))
            G.add_edges_from((tuple(map(int, e.split())) for e in edgelist))
            yield G