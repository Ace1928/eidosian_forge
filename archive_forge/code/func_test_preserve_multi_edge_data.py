import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_preserve_multi_edge_data(self):
    """
        Test that data and keys of edges are preserved on consequent
        write and reads
        """
    G = nx.MultiGraph()
    G.add_node(1)
    G.add_node(2)
    G.add_edges_from([(1, 2), (1, 2, {'key': 'data_key1'}), (1, 2, {'id': 'data_id2'}), (1, 2, {'key': 'data_key3', 'id': 'data_id3'}), (1, 2, 103, {'key': 'data_key4'}), (1, 2, 104, {'id': 'data_id5'}), (1, 2, 105, {'key': 'data_key6', 'id': 'data_id7'})])
    fh = io.BytesIO()
    nx.write_graphml(G, fh)
    fh.seek(0)
    H = nx.read_graphml(fh, node_type=int)
    assert edges_equal(G.edges(data=True, keys=True), H.edges(data=True, keys=True))
    assert G._adj == H._adj
    Gadj = {str(node): {str(nbr): {str(ekey): dd for ekey, dd in key_dict.items()} for nbr, key_dict in nbr_dict.items()} for node, nbr_dict in G._adj.items()}
    fh.seek(0)
    HH = nx.read_graphml(fh, node_type=str, edge_key_type=str)
    assert Gadj == HH._adj
    fh.seek(0)
    string_fh = fh.read()
    HH = nx.parse_graphml(string_fh, node_type=str, edge_key_type=str)
    assert Gadj == HH._adj