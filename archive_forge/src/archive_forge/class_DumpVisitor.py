import ast
import tokenize
from argparse import ArgumentParser, RawTextHelpFormatter
from enum import Enum
from nodedump import debug_format_node
class DumpVisitor(ast.NodeVisitor):

    def __init__(self):
        ast.NodeVisitor.__init__(self)
        self._indent = 0
        self._printed_source_lines = {-1}

    def generic_visit(self, node):
        node_type = get_node_type(node)
        if _opt_verbose and node_type in (NodeType.IGNORE, NodeType.PRINT_ONE_LINE):
            node_type = NodeType.PRINT
        if node_type == NodeType.IGNORE:
            return
        self._indent = self._indent + 1
        indent = '    ' * self._indent
        if node_type == NodeType.PRINT_WITH_SOURCE:
            line_number = node.lineno - 1
            if line_number not in self._printed_source_lines:
                self._printed_source_lines.add(line_number)
                line = _source_lines[line_number]
                non_space = first_non_space(line)
                print('{:04d} {}{}'.format(line_number, '_' * non_space, line[non_space:]))
        if node_type == NodeType.PRINT_ONE_LINE:
            print(indent, debug_format_node(node))
        else:
            print(indent, '>', debug_format_node(node))
            ast.NodeVisitor.generic_visit(self, node)
            print(indent, '<', type(node).__name__)
        self._indent = self._indent - 1