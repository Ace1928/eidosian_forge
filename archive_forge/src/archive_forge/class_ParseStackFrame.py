import sys
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
class ParseStackFrame(NamedTuple):
    node: Any
    indent: int
    has_vertical_child: Union[int, None]