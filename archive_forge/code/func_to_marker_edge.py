from numbers import Number
import networkx as nx
from networkx.drawing.layout import (
def to_marker_edge(marker_size, marker):
    if marker in 's^>v<d':
        return np.sqrt(2 * marker_size) / 2
    else:
        return np.sqrt(marker_size) / 2