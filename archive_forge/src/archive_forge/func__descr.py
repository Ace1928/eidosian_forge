from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
def _descr(self, buf, add_metadata):

    def descr_arg(i, a):
        if i in self.arg_attributes:
            attrs = ' '.join(self.arg_attributes[i]._to_list(a.type)) + ' '
        else:
            attrs = ''
        return '{0} {1}{2}'.format(a.type, attrs, a.get_reference())
    args = ', '.join([descr_arg(i, a) for i, a in enumerate(self.args)])
    fnty = self.callee.function_type
    if fnty.var_arg:
        ty = fnty
    else:
        ty = fnty.return_type
    callee_ref = '{0} {1}'.format(ty, self.callee.get_reference())
    if self.cconv:
        callee_ref = '{0} {1}'.format(self.cconv, callee_ref)
    tail_marker = ''
    if self.tail:
        tail_marker = '{0} '.format(self.tail)
    fn_attrs = ' ' + ' '.join(self.attributes._to_list(fnty.return_type)) if self.attributes else ''
    fm_attrs = ' ' + ' '.join(self.fastmath._to_list(fnty.return_type)) if self.fastmath else ''
    buf.append('{tail}{op}{fastmath} {callee}({args}){attr}{meta}\n'.format(tail=tail_marker, op=self.opname, callee=callee_ref, fastmath=fm_attrs, args=args, attr=fn_attrs, meta=self._stringify_metadata(leading_comma=True) if add_metadata else ''))