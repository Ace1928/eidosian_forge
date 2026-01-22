import sys
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
class AsciiDirectedGlyphs(AsciiBaseGlyphs):
    last: str = 'L-> '
    mid: str = '|-> '
    backedge: str = '<-'
    vertical_edge: str = '!'