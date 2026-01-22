import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def add_graph_element(self, G):
    """
        Serialize graph G in GraphML to the stream.
        """
    if G.is_directed():
        default_edge_type = 'directed'
    else:
        default_edge_type = 'undirected'
    graphid = G.graph.pop('id', None)
    if graphid is None:
        graph_element = self._xml.element('graph', edgedefault=default_edge_type)
    else:
        graph_element = self._xml.element('graph', edgedefault=default_edge_type, id=graphid)
    graphdata = {k: v for k, v in G.graph.items() if k not in ('node_default', 'edge_default')}
    node_default = G.graph.get('node_default', {})
    edge_default = G.graph.get('edge_default', {})
    for k, v in graphdata.items():
        self.attribute_types[str(k), 'graph'].add(type(v))
    for k, v in graphdata.items():
        element_type = self.get_xml_type(self.attr_type(k, 'graph', v))
        self.get_key(str(k), element_type, 'graph', None)
    for node, d in G.nodes(data=True):
        for k, v in d.items():
            self.attribute_types[str(k), 'node'].add(type(v))
    for node, d in G.nodes(data=True):
        for k, v in d.items():
            T = self.get_xml_type(self.attr_type(k, 'node', v))
            self.get_key(str(k), T, 'node', node_default.get(k))
    if G.is_multigraph():
        for u, v, ekey, d in G.edges(keys=True, data=True):
            for k, v in d.items():
                self.attribute_types[str(k), 'edge'].add(type(v))
        for u, v, ekey, d in G.edges(keys=True, data=True):
            for k, v in d.items():
                T = self.get_xml_type(self.attr_type(k, 'edge', v))
                self.get_key(str(k), T, 'edge', edge_default.get(k))
    else:
        for u, v, d in G.edges(data=True):
            for k, v in d.items():
                self.attribute_types[str(k), 'edge'].add(type(v))
        for u, v, d in G.edges(data=True):
            for k, v in d.items():
                T = self.get_xml_type(self.attr_type(k, 'edge', v))
                self.get_key(str(k), T, 'edge', edge_default.get(k))
    for key in self.xml:
        self._xml.write(key, pretty_print=self._prettyprint)
    incremental_writer = IncrementalElement(self._xml, self._prettyprint)
    with graph_element:
        self.add_attributes('graph', incremental_writer, graphdata, {})
        self.add_nodes(G, incremental_writer)
        self.add_edges(G, incremental_writer)