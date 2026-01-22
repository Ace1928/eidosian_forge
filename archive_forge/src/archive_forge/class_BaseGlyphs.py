import sys
import warnings
from collections import defaultdict
import networkx as nx
from networkx.utils import open_file
class BaseGlyphs:

    @classmethod
    def as_dict(cls):
        return {a: getattr(cls, a) for a in dir(cls) if not a.startswith('_') and a != 'as_dict'}