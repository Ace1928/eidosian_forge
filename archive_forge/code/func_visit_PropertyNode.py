from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def visit_PropertyNode(self, node):
    if not self.current_directives['embedsignature']:
        return node
    self._setup_format()
    entry = node.entry
    body = node.body
    prop_name = entry.name
    type_name = None
    if entry.visibility == 'public':
        if self.is_format_c:
            type_name = entry.type.declaration_code('', for_display=1)
            if not entry.type.is_pyobject:
                type_name = "'%s'" % type_name
            elif entry.type.is_extension_type:
                type_name = entry.type.module_name + '.' + type_name
        elif self.is_format_python:
            type_name = self._fmt_type(entry.type)
    if type_name is None:
        for stat in body.stats:
            if stat.name != '__get__':
                continue
            if self.is_format_c:
                prop_name = '%s.%s' % (self.class_name, prop_name)
            ret_annotation = stat.return_type_annotation
            if ret_annotation:
                type_name = self._fmt_annotation(ret_annotation)
    if type_name is not None:
        signature = '%s: %s' % (prop_name, type_name)
        new_doc = self._embed_signature(signature, entry.doc)
        if not self.is_format_clinic:
            entry.doc = EncodedString(new_doc)
    return node