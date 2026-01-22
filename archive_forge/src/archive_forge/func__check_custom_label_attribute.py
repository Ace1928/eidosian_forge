from itertools import chain
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def _check_custom_label_attribute(input_trees, res_tree, label_attribute):
    res_attr_dict = nx.get_node_attributes(res_tree, label_attribute)
    res_attr_set = set(res_attr_dict.values())
    input_label = (tree for tree, root in input_trees)
    input_label_set = set(chain.from_iterable(input_label))
    return res_attr_set == input_label_set