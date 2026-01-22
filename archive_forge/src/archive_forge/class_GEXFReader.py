import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
class GEXFReader(GEXF):

    def __init__(self, node_type=None, version='1.2draft'):
        self.construct_types()
        self.node_type = node_type
        self.simple_graph = True
        self.set_version(version)

    def __call__(self, stream):
        self.xml = ElementTree(file=stream)
        g = self.xml.find(f'{{{self.NS_GEXF}}}graph')
        if g is not None:
            return self.make_graph(g)
        for version in self.versions:
            self.set_version(version)
            g = self.xml.find(f'{{{self.NS_GEXF}}}graph')
            if g is not None:
                return self.make_graph(g)
        raise nx.NetworkXError('No <graph> element in GEXF file.')

    def make_graph(self, graph_xml):
        edgedefault = graph_xml.get('defaultedgetype', None)
        if edgedefault == 'directed':
            G = nx.MultiDiGraph()
        else:
            G = nx.MultiGraph()
        graph_name = graph_xml.get('name', '')
        if graph_name != '':
            G.graph['name'] = graph_name
        graph_start = graph_xml.get('start')
        if graph_start is not None:
            G.graph['start'] = graph_start
        graph_end = graph_xml.get('end')
        if graph_end is not None:
            G.graph['end'] = graph_end
        graph_mode = graph_xml.get('mode', '')
        if graph_mode == 'dynamic':
            G.graph['mode'] = 'dynamic'
        else:
            G.graph['mode'] = 'static'
        self.timeformat = graph_xml.get('timeformat')
        if self.timeformat == 'date':
            self.timeformat = 'string'
        attributes_elements = graph_xml.findall(f'{{{self.NS_GEXF}}}attributes')
        node_attr = {}
        node_default = {}
        edge_attr = {}
        edge_default = {}
        for a in attributes_elements:
            attr_class = a.get('class')
            if attr_class == 'node':
                na, nd = self.find_gexf_attributes(a)
                node_attr.update(na)
                node_default.update(nd)
                G.graph['node_default'] = node_default
            elif attr_class == 'edge':
                ea, ed = self.find_gexf_attributes(a)
                edge_attr.update(ea)
                edge_default.update(ed)
                G.graph['edge_default'] = edge_default
            else:
                raise
        ea = {'weight': {'type': 'double', 'mode': 'static', 'title': 'weight'}}
        ed = {}
        edge_attr.update(ea)
        edge_default.update(ed)
        G.graph['edge_default'] = edge_default
        nodes_element = graph_xml.find(f'{{{self.NS_GEXF}}}nodes')
        if nodes_element is not None:
            for node_xml in nodes_element.findall(f'{{{self.NS_GEXF}}}node'):
                self.add_node(G, node_xml, node_attr)
        edges_element = graph_xml.find(f'{{{self.NS_GEXF}}}edges')
        if edges_element is not None:
            for edge_xml in edges_element.findall(f'{{{self.NS_GEXF}}}edge'):
                self.add_edge(G, edge_xml, edge_attr)
        if self.simple_graph:
            if G.is_directed():
                G = nx.DiGraph(G)
            else:
                G = nx.Graph(G)
        return G

    def add_node(self, G, node_xml, node_attr, node_pid=None):
        data = self.decode_attr_elements(node_attr, node_xml)
        data = self.add_parents(data, node_xml)
        if self.VERSION == '1.1':
            data = self.add_slices(data, node_xml)
        else:
            data = self.add_spells(data, node_xml)
        data = self.add_viz(data, node_xml)
        data = self.add_start_end(data, node_xml)
        node_id = node_xml.get('id')
        if self.node_type is not None:
            node_id = self.node_type(node_id)
        node_label = node_xml.get('label')
        data['label'] = node_label
        node_pid = node_xml.get('pid', node_pid)
        if node_pid is not None:
            data['pid'] = node_pid
        subnodes = node_xml.find(f'{{{self.NS_GEXF}}}nodes')
        if subnodes is not None:
            for node_xml in subnodes.findall(f'{{{self.NS_GEXF}}}node'):
                self.add_node(G, node_xml, node_attr, node_pid=node_id)
        G.add_node(node_id, **data)

    def add_start_end(self, data, xml):
        ttype = self.timeformat
        node_start = xml.get('start')
        if node_start is not None:
            data['start'] = self.python_type[ttype](node_start)
        node_end = xml.get('end')
        if node_end is not None:
            data['end'] = self.python_type[ttype](node_end)
        return data

    def add_viz(self, data, node_xml):
        viz = {}
        color = node_xml.find(f'{{{self.NS_VIZ}}}color')
        if color is not None:
            if self.VERSION == '1.1':
                viz['color'] = {'r': int(color.get('r')), 'g': int(color.get('g')), 'b': int(color.get('b'))}
            else:
                viz['color'] = {'r': int(color.get('r')), 'g': int(color.get('g')), 'b': int(color.get('b')), 'a': float(color.get('a', 1))}
        size = node_xml.find(f'{{{self.NS_VIZ}}}size')
        if size is not None:
            viz['size'] = float(size.get('value'))
        thickness = node_xml.find(f'{{{self.NS_VIZ}}}thickness')
        if thickness is not None:
            viz['thickness'] = float(thickness.get('value'))
        shape = node_xml.find(f'{{{self.NS_VIZ}}}shape')
        if shape is not None:
            viz['shape'] = shape.get('shape')
            if viz['shape'] == 'image':
                viz['shape'] = shape.get('uri')
        position = node_xml.find(f'{{{self.NS_VIZ}}}position')
        if position is not None:
            viz['position'] = {'x': float(position.get('x', 0)), 'y': float(position.get('y', 0)), 'z': float(position.get('z', 0))}
        if len(viz) > 0:
            data['viz'] = viz
        return data

    def add_parents(self, data, node_xml):
        parents_element = node_xml.find(f'{{{self.NS_GEXF}}}parents')
        if parents_element is not None:
            data['parents'] = []
            for p in parents_element.findall(f'{{{self.NS_GEXF}}}parent'):
                parent = p.get('for')
                data['parents'].append(parent)
        return data

    def add_slices(self, data, node_or_edge_xml):
        slices_element = node_or_edge_xml.find(f'{{{self.NS_GEXF}}}slices')
        if slices_element is not None:
            data['slices'] = []
            for s in slices_element.findall(f'{{{self.NS_GEXF}}}slice'):
                start = s.get('start')
                end = s.get('end')
                data['slices'].append((start, end))
        return data

    def add_spells(self, data, node_or_edge_xml):
        spells_element = node_or_edge_xml.find(f'{{{self.NS_GEXF}}}spells')
        if spells_element is not None:
            data['spells'] = []
            ttype = self.timeformat
            for s in spells_element.findall(f'{{{self.NS_GEXF}}}spell'):
                start = self.python_type[ttype](s.get('start'))
                end = self.python_type[ttype](s.get('end'))
                data['spells'].append((start, end))
        return data

    def add_edge(self, G, edge_element, edge_attr):
        edge_direction = edge_element.get('type')
        if G.is_directed() and edge_direction == 'undirected':
            raise nx.NetworkXError('Undirected edge found in directed graph.')
        if not G.is_directed() and edge_direction == 'directed':
            raise nx.NetworkXError('Directed edge found in undirected graph.')
        source = edge_element.get('source')
        target = edge_element.get('target')
        if self.node_type is not None:
            source = self.node_type(source)
            target = self.node_type(target)
        data = self.decode_attr_elements(edge_attr, edge_element)
        data = self.add_start_end(data, edge_element)
        if self.VERSION == '1.1':
            data = self.add_slices(data, edge_element)
        else:
            data = self.add_spells(data, edge_element)
        edge_id = edge_element.get('id')
        if edge_id is not None:
            data['id'] = edge_id
        multigraph_key = data.pop('networkx_key', None)
        if multigraph_key is not None:
            edge_id = multigraph_key
        weight = edge_element.get('weight')
        if weight is not None:
            data['weight'] = float(weight)
        edge_label = edge_element.get('label')
        if edge_label is not None:
            data['label'] = edge_label
        if G.has_edge(source, target):
            self.simple_graph = False
        G.add_edge(source, target, key=edge_id, **data)
        if edge_direction == 'mutual':
            G.add_edge(target, source, key=edge_id, **data)

    def decode_attr_elements(self, gexf_keys, obj_xml):
        attr = {}
        attr_element = obj_xml.find(f'{{{self.NS_GEXF}}}attvalues')
        if attr_element is not None:
            for a in attr_element.findall(f'{{{self.NS_GEXF}}}attvalue'):
                key = a.get('for')
                try:
                    title = gexf_keys[key]['title']
                except KeyError as err:
                    raise nx.NetworkXError(f'No attribute defined for={key}.') from err
                atype = gexf_keys[key]['type']
                value = a.get('value')
                if atype == 'boolean':
                    value = self.convert_bool[value]
                else:
                    value = self.python_type[atype](value)
                if gexf_keys[key]['mode'] == 'dynamic':
                    ttype = self.timeformat
                    start = self.python_type[ttype](a.get('start'))
                    end = self.python_type[ttype](a.get('end'))
                    if title in attr:
                        attr[title].append((value, start, end))
                    else:
                        attr[title] = [(value, start, end)]
                else:
                    attr[title] = value
        return attr

    def find_gexf_attributes(self, attributes_element):
        attrs = {}
        defaults = {}
        mode = attributes_element.get('mode')
        for k in attributes_element.findall(f'{{{self.NS_GEXF}}}attribute'):
            attr_id = k.get('id')
            title = k.get('title')
            atype = k.get('type')
            attrs[attr_id] = {'title': title, 'type': atype, 'mode': mode}
            default = k.find(f'{{{self.NS_GEXF}}}default')
            if default is not None:
                if atype == 'boolean':
                    value = self.convert_bool[default.text]
                else:
                    value = self.python_type[atype](default.text)
                defaults[title] = value
        return (attrs, defaults)