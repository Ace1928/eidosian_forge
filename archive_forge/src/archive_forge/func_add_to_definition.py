from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
@staticmethod
def add_to_definition(definition, name, dnode_or_dnodes):
    if isinstance(dnode_or_dnodes, Def):
        definition[name].add(dnode_or_dnodes)
    else:
        definition[name].update(dnode_or_dnodes)