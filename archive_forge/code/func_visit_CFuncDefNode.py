from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def visit_CFuncDefNode(self, node):
    if not node.overridable:
        return node
    if not self.current_directives['embedsignature']:
        return node
    self._setup_format()
    signature = self._fmt_signature(self.class_name, node.declarator.base.name, node.declarator.args, return_type=node.return_type)
    if signature:
        if node.entry.doc is not None:
            old_doc = node.entry.doc
        elif getattr(node, 'py_func', None) is not None:
            old_doc = node.py_func.entry.doc
        else:
            old_doc = None
        new_doc = self._embed_signature(signature, old_doc)
        node.entry.doc = EncodedString(new_doc)
        py_func = getattr(node, 'py_func', None)
        if py_func is not None:
            py_func.entry.doc = EncodedString(new_doc)
    return node