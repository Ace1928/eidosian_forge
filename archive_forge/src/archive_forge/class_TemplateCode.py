from collections import OrderedDict
from textwrap import dedent
import operator
from . import ExprNodes
from . import Nodes
from . import PyrexTypes
from . import Builtin
from . import Naming
from .Errors import error, warning
from .Code import UtilityCode, TempitaUtilityCode, PyxCodeWriter
from .Visitor import VisitorTransform
from .StringEncoding import EncodedString
from .TreeFragment import TreeFragment
from .ParseTreeTransforms import NormalizeTree, SkipDeclarations
from .Options import copy_inherited_directives
class TemplateCode(object):
    """
    Adds the ability to keep track of placeholder argument names to PyxCodeWriter.

    Also adds extra_stats which are nodes bundled at the end when this
    is converted to a tree.
    """
    _placeholder_count = 0

    def __init__(self, writer=None, placeholders=None, extra_stats=None):
        self.writer = PyxCodeWriter() if writer is None else writer
        self.placeholders = {} if placeholders is None else placeholders
        self.extra_stats = [] if extra_stats is None else extra_stats

    def add_code_line(self, code_line):
        self.writer.putln(code_line)

    def add_code_lines(self, code_lines):
        for line in code_lines:
            self.writer.putln(line)

    def reset(self):
        self.writer.reset()

    def empty(self):
        return self.writer.empty()

    def indenter(self):
        return self.writer.indenter()

    def new_placeholder(self, field_names, value):
        name = self._new_placeholder_name(field_names)
        self.placeholders[name] = value
        return name

    def add_extra_statements(self, statements):
        if self.extra_stats is None:
            assert False, 'Can only use add_extra_statements on top-level writer'
        self.extra_stats.extend(statements)

    def _new_placeholder_name(self, field_names):
        while True:
            name = 'DATACLASS_PLACEHOLDER_%d' % self._placeholder_count
            if name not in self.placeholders and name not in field_names:
                break
            self._placeholder_count += 1
        return name

    def generate_tree(self, level='c_class'):
        stat_list_node = TreeFragment(self.writer.getvalue(), level=level, pipeline=[NormalizeTree(None)]).substitute(self.placeholders)
        stat_list_node.stats += self.extra_stats
        return stat_list_node

    def insertion_point(self):
        new_writer = self.writer.insertion_point()
        return TemplateCode(writer=new_writer, placeholders=self.placeholders, extra_stats=self.extra_stats)