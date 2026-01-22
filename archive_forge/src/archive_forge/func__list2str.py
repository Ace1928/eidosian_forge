import abc
from cliff.formatters import base
from networkx.drawing.nx_pydot import write_dot
from networkx.readwrite.graphml import GraphMLWriter
from networkx.readwrite import json_graph
import networkx as nx
@staticmethod
def _list2str(node):
    for k, v in node.items():
        if type(v) == list:
            node[k] = str(v)
        if type(v) == str and ':' in v:
            node[k] = '"' + v + '"'