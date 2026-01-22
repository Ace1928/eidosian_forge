from __future__ import absolute_import
import cython
from . import Builtin
from . import ExprNodes
from . import Nodes
from . import Options
from . import PyrexTypes
from .Visitor import TreeVisitor, CythonTransform
from .Errors import error, warning, InternalError
class GV(object):
    """Graphviz DOT renderer."""

    def __init__(self, name, flow):
        self.name = name
        self.flow = flow

    def render(self, fp, ctx, annotate_defs=False):
        fp.write(' subgraph %s {\n' % self.name)
        for block in self.flow.blocks:
            label = ctx.extract_sources(block)
            if annotate_defs:
                for stat in block.stats:
                    if isinstance(stat, NameAssignment):
                        label += '\n %s [%s %s]' % (stat.entry.name, 'deletion' if stat.is_deletion else 'definition', stat.pos[1])
                    elif isinstance(stat, NameReference):
                        if stat.entry:
                            label += '\n %s [reference %s]' % (stat.entry.name, stat.pos[1])
            if not label:
                label = 'empty'
            pid = ctx.nodeid(block)
            fp.write('  %s [label="%s"];\n' % (pid, ctx.escape(label)))
        for block in self.flow.blocks:
            pid = ctx.nodeid(block)
            for child in block.children:
                fp.write('  %s -> %s;\n' % (pid, ctx.nodeid(child)))
        fp.write(' }\n')