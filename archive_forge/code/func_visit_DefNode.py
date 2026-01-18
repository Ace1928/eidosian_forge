from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def visit_DefNode(self, node):
    if not self.current_directives['embedsignature']:
        return node
    self._setup_format()
    is_constructor = False
    hide_self = False
    if node.entry.is_special:
        is_constructor = self.class_node and node.name == '__init__'
        if not is_constructor:
            return node
        class_name = None
        func_name = node.name
        if self.is_format_c:
            func_name = self.class_name
            hide_self = True
    else:
        class_name, func_name = (self.class_name, node.name)
    npoargs = getattr(node, 'num_posonly_args', 0)
    nkargs = getattr(node, 'num_kwonly_args', 0)
    npargs = len(node.args) - nkargs - npoargs
    signature = self._fmt_signature(class_name, func_name, node.args, npoargs, npargs, node.star_arg, nkargs, node.starstar_arg, return_expr=node.return_type_annotation, return_type=None, hide_self=hide_self)
    if signature:
        if is_constructor and self.is_format_c:
            doc_holder = self.class_node.entry.type.scope
        else:
            doc_holder = node.entry
        if doc_holder.doc is not None:
            old_doc = doc_holder.doc
        elif not is_constructor and getattr(node, 'py_func', None) is not None:
            old_doc = node.py_func.entry.doc
        else:
            old_doc = None
        new_doc = self._embed_signature(signature, old_doc)
        doc_holder.doc = EncodedString(new_doc)
        if not is_constructor and getattr(node, 'py_func', None) is not None:
            node.py_func.entry.doc = EncodedString(new_doc)
    return node