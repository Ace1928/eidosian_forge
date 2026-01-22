from llvmlite.ir import types
from llvmlite.ir.values import (Block, Function, Value, NamedValue, Constant,
from llvmlite.ir._utils import _HasMetadata
class CallInstr(Instruction):

    def __init__(self, parent, func, args, name='', cconv=None, tail=None, fastmath=(), attrs=(), arg_attrs=None):
        self.cconv = func.calling_convention if cconv is None and isinstance(func, Function) else cconv
        if isinstance(tail, str) and tail in TailMarkerOptions:
            pass
        elif tail:
            tail = 'tail'
        else:
            tail = ''
        self.tail = tail
        self.fastmath = FastMathFlags(fastmath)
        self.attributes = CallInstrAttributes(attrs)
        self.arg_attributes = {}
        if arg_attrs:
            for idx, attrs in arg_attrs.items():
                if not 0 <= idx < len(args):
                    raise ValueError('Invalid argument index {}'.format(idx))
                self.arg_attributes[idx] = ArgumentAttributes(attrs)
        args = list(args)
        for i in range(len(func.function_type.args)):
            arg = args[i]
            expected_type = func.function_type.args[i]
            if isinstance(expected_type, types.MetaDataType) and arg.type != expected_type:
                arg = MetaDataArgument(arg)
            if arg.type != expected_type:
                msg = 'Type of #{0} arg mismatch: {1} != {2}'.format(1 + i, expected_type, arg.type)
                raise TypeError(msg)
            args[i] = arg
        super(CallInstr, self).__init__(parent, func.function_type.return_type, 'call', [func] + list(args), name=name)

    @property
    def callee(self):
        return self.operands[0]

    @callee.setter
    def callee(self, newcallee):
        self.operands[0] = newcallee

    @property
    def args(self):
        return self.operands[1:]

    def replace_callee(self, newfunc):
        if newfunc.function_type != self.callee.function_type:
            raise TypeError('New function has incompatible type')
        self.callee = newfunc

    @property
    def called_function(self):
        """The callee function"""
        return self.callee

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

    def descr(self, buf):
        self._descr(buf, add_metadata=True)