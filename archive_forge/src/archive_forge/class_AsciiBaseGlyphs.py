import sys
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
class AsciiBaseGlyphs(BaseGlyphs):
    empty: str = '+'
    newtree_last: str = '+-- '
    newtree_mid: str = '+-- '
    endof_forest: str = '    '
    within_forest: str = ':   '
    within_tree: str = '|   '