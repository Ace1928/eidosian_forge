import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
class GraphMLReader(GraphML):
    """Read a GraphML document.  Produces NetworkX graph objects."""

    def __init__(self, node_type=str, edge_key_type=int, force_multigraph=False):
        self.construct_types()
        self.node_type = node_type
        self.edge_key_type = edge_key_type
        self.multigraph = force_multigraph
        self.edge_ids = {}

    def __call__(self, path=None, string=None):
        from xml.etree.ElementTree import ElementTree, fromstring
        if path is not None:
            self.xml = ElementTree(file=path)
        elif string is not None:
            self.xml = fromstring(string)
        else:
            raise ValueError("Must specify either 'path' or 'string' as kwarg")
        keys, defaults = self.find_graphml_keys(self.xml)
        for g in self.xml.findall(f'{{{self.NS_GRAPHML}}}graph'):
            yield self.make_graph(g, keys, defaults)

    def make_graph(self, graph_xml, graphml_keys, defaults, G=None):
        edgedefault = graph_xml.get('edgedefault', None)
        if G is None:
            if edgedefault == 'directed':
                G = nx.MultiDiGraph()
            else:
                G = nx.MultiGraph()
        G.graph['node_default'] = {}
        G.graph['edge_default'] = {}
        for key_id, value in defaults.items():
            key_for = graphml_keys[key_id]['for']
            name = graphml_keys[key_id]['name']
            python_type = graphml_keys[key_id]['type']
            if key_for == 'node':
                G.graph['node_default'].update({name: python_type(value)})
            if key_for == 'edge':
                G.graph['edge_default'].update({name: python_type(value)})
        hyperedge = graph_xml.find(f'{{{self.NS_GRAPHML}}}hyperedge')
        if hyperedge is not None:
            raise nx.NetworkXError("GraphML reader doesn't support hyperedges")
        for node_xml in graph_xml.findall(f'{{{self.NS_GRAPHML}}}node'):
            self.add_node(G, node_xml, graphml_keys, defaults)
        for edge_xml in graph_xml.findall(f'{{{self.NS_GRAPHML}}}edge'):
            self.add_edge(G, edge_xml, graphml_keys)
        data = self.decode_data_elements(graphml_keys, graph_xml)
        G.graph.update(data)
        if self.multigraph:
            return G
        G = nx.DiGraph(G) if G.is_directed() else nx.Graph(G)
        nx.set_edge_attributes(G, values=self.edge_ids, name='id')
        return G

    def add_node(self, G, node_xml, graphml_keys, defaults):
        """Add a node to the graph."""
        ports = node_xml.find(f'{{{self.NS_GRAPHML}}}port')
        if ports is not None:
            warnings.warn('GraphML port tag not supported.')
        node_id = self.node_type(node_xml.get('id'))
        data = self.decode_data_elements(graphml_keys, node_xml)
        G.add_node(node_id, **data)
        if node_xml.attrib.get('yfiles.foldertype') == 'group':
            graph_xml = node_xml.find(f'{{{self.NS_GRAPHML}}}graph')
            self.make_graph(graph_xml, graphml_keys, defaults, G)

    def add_edge(self, G, edge_element, graphml_keys):
        """Add an edge to the graph."""
        ports = edge_element.find(f'{{{self.NS_GRAPHML}}}port')
        if ports is not None:
            warnings.warn('GraphML port tag not supported.')
        directed = edge_element.get('directed')
        if G.is_directed() and directed == 'false':
            msg = 'directed=false edge found in directed graph.'
            raise nx.NetworkXError(msg)
        if not G.is_directed() and directed == 'true':
            msg = 'directed=true edge found in undirected graph.'
            raise nx.NetworkXError(msg)
        source = self.node_type(edge_element.get('source'))
        target = self.node_type(edge_element.get('target'))
        data = self.decode_data_elements(graphml_keys, edge_element)
        edge_id = edge_element.get('id')
        if edge_id:
            self.edge_ids[source, target] = edge_id
            try:
                edge_id = self.edge_key_type(edge_id)
            except ValueError:
                pass
        else:
            edge_id = data.get('key')
        if G.has_edge(source, target):
            self.multigraph = True
        G.add_edges_from([(source, target, edge_id, data)])

    def decode_data_elements(self, graphml_keys, obj_xml):
        """Use the key information to decode the data XML if present."""
        data = {}
        for data_element in obj_xml.findall(f'{{{self.NS_GRAPHML}}}data'):
            key = data_element.get('key')
            try:
                data_name = graphml_keys[key]['name']
                data_type = graphml_keys[key]['type']
            except KeyError as err:
                raise nx.NetworkXError(f'Bad GraphML data: no key {key}') from err
            text = data_element.text
            if text is not None and len(list(data_element)) == 0:
                if data_type == bool:
                    data[data_name] = self.convert_bool[text.lower()]
                else:
                    data[data_name] = data_type(text)
            elif len(list(data_element)) > 0:
                node_label = None
                gn = data_element.find(f'{{{self.NS_Y}}}GenericNode')
                if gn:
                    data['shape_type'] = gn.get('configuration')
                for node_type in ['GenericNode', 'ShapeNode', 'SVGNode', 'ImageNode']:
                    pref = f'{{{self.NS_Y}}}{node_type}/{{{self.NS_Y}}}'
                    geometry = data_element.find(f'{pref}Geometry')
                    if geometry is not None:
                        data['x'] = geometry.get('x')
                        data['y'] = geometry.get('y')
                    if node_label is None:
                        node_label = data_element.find(f'{pref}NodeLabel')
                    shape = data_element.find(f'{pref}Shape')
                    if shape is not None:
                        data['shape_type'] = shape.get('type')
                if node_label is not None:
                    data['label'] = node_label.text
                for edge_type in ['PolyLineEdge', 'SplineEdge', 'QuadCurveEdge', 'BezierEdge', 'ArcEdge']:
                    pref = f'{{{self.NS_Y}}}{edge_type}/{{{self.NS_Y}}}'
                    edge_label = data_element.find(f'{pref}EdgeLabel')
                    if edge_label is not None:
                        break
                if edge_label is not None:
                    data['label'] = edge_label.text
        return data

    def find_graphml_keys(self, graph_element):
        """Extracts all the keys and key defaults from the xml."""
        graphml_keys = {}
        graphml_key_defaults = {}
        for k in graph_element.findall(f'{{{self.NS_GRAPHML}}}key'):
            attr_id = k.get('id')
            attr_type = k.get('attr.type')
            attr_name = k.get('attr.name')
            yfiles_type = k.get('yfiles.type')
            if yfiles_type is not None:
                attr_name = yfiles_type
                attr_type = 'yfiles'
            if attr_type is None:
                attr_type = 'string'
                warnings.warn(f'No key type for id {attr_id}. Using string')
            if attr_name is None:
                raise nx.NetworkXError(f'Unknown key for id {attr_id}.')
            graphml_keys[attr_id] = {'name': attr_name, 'type': self.python_type[attr_type], 'for': k.get('for')}
            default = k.find(f'{{{self.NS_GRAPHML}}}default')
            if default is not None:
                python_type = graphml_keys[attr_id]['type']
                if python_type == bool:
                    graphml_key_defaults[attr_id] = self.convert_bool[default.text.lower()]
                else:
                    graphml_key_defaults[attr_id] = python_type(default.text)
        return (graphml_keys, graphml_key_defaults)