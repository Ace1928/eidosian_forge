import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
def add_attributes(self, scope, xml_obj, data, default):
    """Appends attribute data."""
    for k, v in data.items():
        data_element = self.add_data(str(k), self.attr_type(str(k), scope, v), str(v), scope, default.get(k))
        xml_obj.append(data_element)