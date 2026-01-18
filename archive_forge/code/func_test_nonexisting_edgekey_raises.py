import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_nonexisting_edgekey_raises(self):
    with pytest.raises(nx.exception.NetworkXError):
        nx.from_pandas_edgelist(self.df, source='source', target='target', edge_key='Not_real', edge_attr=True, create_using=nx.MultiGraph())