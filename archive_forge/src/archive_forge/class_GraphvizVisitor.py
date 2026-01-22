import operator
from jmespath import functions
from jmespath.compat import string_type
from numbers import Number
class GraphvizVisitor(Visitor):

    def __init__(self):
        super(GraphvizVisitor, self).__init__()
        self._lines = []
        self._count = 1

    def visit(self, node, *args, **kwargs):
        self._lines.append('digraph AST {')
        current = '%s%s' % (node['type'], self._count)
        self._count += 1
        self._visit(node, current)
        self._lines.append('}')
        return '\n'.join(self._lines)

    def _visit(self, node, current):
        self._lines.append('%s [label="%s(%s)"]' % (current, node['type'], node.get('value', '')))
        for child in node.get('children', []):
            child_name = '%s%s' % (child['type'], self._count)
            self._count += 1
            self._lines.append('  %s -> %s' % (current, child_name))
            self._visit(child, child_name)