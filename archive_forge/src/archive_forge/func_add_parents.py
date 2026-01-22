import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
def add_parents(self, data, node_xml):
    parents_element = node_xml.find(f'{{{self.NS_GEXF}}}parents')
    if parents_element is not None:
        data['parents'] = []
        for p in parents_element.findall(f'{{{self.NS_GEXF}}}parent'):
            parent = p.get('for')
            data['parents'].append(parent)
    return data