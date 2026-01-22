import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def add_edges(self, G, graph_element):
    if G.is_multigraph():
        for u, v, key, data in G.edges(data=True, keys=True):
            edge_element = self.myElement('edge', source=str(u), target=str(v), id=str(data.get(self.edge_id_from_attribute)) if self.edge_id_from_attribute and self.edge_id_from_attribute in data else str(key))
            default = G.graph.get('edge_default', {})
            self.add_attributes('edge', edge_element, data, default)
            graph_element.append(edge_element)
    else:
        for u, v, data in G.edges(data=True):
            if self.edge_id_from_attribute and self.edge_id_from_attribute in data:
                edge_element = self.myElement('edge', source=str(u), target=str(v), id=str(data.get(self.edge_id_from_attribute)))
            else:
                edge_element = self.myElement('edge', source=str(u), target=str(v))
            default = G.graph.get('edge_default', {})
            self.add_attributes('edge', edge_element, data, default)
            graph_element.append(edge_element)