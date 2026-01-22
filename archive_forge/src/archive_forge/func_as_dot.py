import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def as_dot(self):
    """Print CFG in DOT format."""
    result = 'digraph CFG {\n'
    for node in self.index.values():
        result += '  %s [label="%s"];\n' % (id(node), node)
    for node in self.index.values():
        for next_ in node.next:
            result += '  %s -> %s;\n' % (id(node), id(next_))
    result += '}'
    return result