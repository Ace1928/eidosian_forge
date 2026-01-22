from __future__ import absolute_import
import re
from io import StringIO
from .Scanning import PyrexScanner, StringSourceDescriptor
from .Symtab import ModuleScope
from . import PyrexTypes
from .Visitor import VisitorTransform
from .Nodes import Node, StatListNode
from .ExprNodes import NameNode
from .StringEncoding import _unicode
from . import Parsing
from . import Main
from . import UtilNodes
class TreeFragment(object):

    def __init__(self, code, name=None, pxds=None, temps=None, pipeline=None, level=None, initial_pos=None):
        if pxds is None:
            pxds = {}
        if temps is None:
            temps = []
        if pipeline is None:
            pipeline = []
        if not name:
            name = '(tree fragment)'
        if isinstance(code, _unicode):

            def fmt(x):
                return u'\n'.join(strip_common_indent(x.split(u'\n')))
            fmt_code = fmt(code)
            fmt_pxds = {}
            for key, value in pxds.items():
                fmt_pxds[key] = fmt(value)
            mod = t = parse_from_strings(name, fmt_code, fmt_pxds, level=level, initial_pos=initial_pos)
            if level is None:
                t = t.body
            if not isinstance(t, StatListNode):
                t = StatListNode(pos=mod.pos, stats=[t])
            for transform in pipeline:
                if transform is None:
                    continue
                t = transform(t)
            self.root = t
        elif isinstance(code, Node):
            if pxds:
                raise NotImplementedError()
            self.root = code
        else:
            raise ValueError('Unrecognized code format (accepts unicode and Node)')
        self.temps = temps

    def copy(self):
        return copy_code_tree(self.root)

    def substitute(self, nodes=None, temps=None, pos=None):
        if nodes is None:
            nodes = {}
        if temps is None:
            temps = []
        return TemplateTransform()(self.root, substitutions=nodes, temps=self.temps + temps, pos=pos)