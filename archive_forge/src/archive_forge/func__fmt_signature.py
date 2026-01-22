from __future__ import absolute_import, print_function
from .Visitor import CythonTransform
from .StringEncoding import EncodedString
from . import Options
from . import PyrexTypes
from ..CodeWriter import ExpressionWriter
from .Errors import warning
def _fmt_signature(self, cls_name, func_name, args, npoargs=0, npargs=0, pargs=None, nkargs=0, kargs=None, return_expr=None, return_type=None, hide_self=False):
    arglist = self._fmt_arglist(args, npoargs, npargs, pargs, nkargs, kargs, hide_self=hide_self)
    arglist_doc = ', '.join(arglist)
    func_doc = '%s(%s)' % (func_name, arglist_doc)
    if self.is_format_c and cls_name:
        func_doc = '%s.%s' % (cls_name, func_doc)
    if not self.is_format_clinic:
        ret_doc = None
        if return_expr:
            ret_doc = self._fmt_annotation(return_expr)
        elif return_type:
            ret_doc = self._fmt_type(return_type)
        if ret_doc:
            func_doc = '%s -> %s' % (func_doc, ret_doc)
    return func_doc