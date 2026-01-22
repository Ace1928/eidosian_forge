import abc
from cliff.formatters import base
from networkx.drawing.nx_pydot import write_dot
from networkx.readwrite.graphml import GraphMLWriter
from networkx.readwrite import json_graph
import networkx as nx
class GraphFormatter(base.SingleFormatter, metaclass=abc.ABCMeta):

    def add_argument_group(self, parser):
        pass

    def emit_one(self, column_names, data, stdout, _=None):
        data = {n: i for n, i in zip(column_names, data)}
        self._reformat(data)
        if nx.__version__ >= '2.0':
            graph = json_graph.node_link_graph(data, attrs={'name': 'graph_index'})
        else:
            graph = json_graph.node_link_graph(data)
        self._write_format(graph, stdout)

    @staticmethod
    def _reformat(data):
        for node in data['nodes']:
            name = node.pop('name', None)
            v_type = node['vitrage_type']
            if name and name != v_type:
                node['label'] = name + '\n' + v_type
            else:
                node['label'] = v_type
            GraphFormatter._list2str(node)
        data['multigraph'] = False
        for node in data['links']:
            node['label'] = node.pop('relationship_type')
            node.pop('key')

    @staticmethod
    def _list2str(node):
        for k, v in node.items():
            if type(v) == list:
                node[k] = str(v)
            if type(v) == str and ':' in v:
                node[k] = '"' + v + '"'

    @abc.abstractmethod
    def _write_format(self, graph, stdout):
        pass