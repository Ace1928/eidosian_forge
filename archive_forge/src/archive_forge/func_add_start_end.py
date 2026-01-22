import itertools
import time
from xml.etree.ElementTree import (
import networkx as nx
from networkx.utils import open_file
def add_start_end(self, data, xml):
    ttype = self.timeformat
    node_start = xml.get('start')
    if node_start is not None:
        data['start'] = self.python_type[ttype](node_start)
    node_end = xml.get('end')
    if node_end is not None:
        data['end'] = self.python_type[ttype](node_end)
    return data