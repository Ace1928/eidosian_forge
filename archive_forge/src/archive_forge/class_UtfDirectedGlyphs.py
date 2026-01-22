import sys
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
class UtfDirectedGlyphs(UtfBaseGlyphs):
    last: str = '└─╼ '
    mid: str = '├─╼ '
    backedge: str = '╾'
    vertical_edge: str = '╽'