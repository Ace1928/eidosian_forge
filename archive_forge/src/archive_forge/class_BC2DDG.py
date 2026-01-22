import os
from dataclasses import dataclass, replace, field, fields
import dis
import operator
from functools import reduce
from typing import (
from collections import ChainMap
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
from numba_rvsdg.rendering.rendering import ByteFlowRenderer
from numba_rvsdg.core.datastructures import block_names
from numba.core.utils import MutableSortedSet, MutableSortedMap
from .regionpasses import (
class BC2DDG:
    stack: list[ValueState]
    effect: ValueState
    in_effect: ValueState
    varmap: dict[str, ValueState]
    incoming_vars: dict[str, ValueState]
    incoming_stackvars: list[ValueState]
    _kw_names: ValueState | None

    def __init__(self):
        self.stack = []
        start_env = Op('start', bc_inst=None)
        self.effect = start_env.add_output('env', is_effect=True)
        self.in_effect = self.effect
        self.varmap = {}
        self.incoming_vars = {}
        self.incoming_stackvars = []
        self._kw_names = None

    def push(self, val: ValueState):
        self.stack.append(val)

    def pop(self) -> ValueState:
        if not self.stack:
            op = Op(opname='stack.incoming', bc_inst=None)
            vs = op.add_output(f'stack.{len(self.incoming_stackvars)}')
            self.stack.append(vs)
            self.incoming_stackvars.append(vs)
        return self.stack.pop()

    def top(self) -> ValueState:
        tos = self.pop()
        self.push(tos)
        return tos

    def _decorate_varname(self, varname: str) -> str:
        return f'var.{varname}'

    def store(self, varname: str, value: ValueState):
        self.varmap[varname] = value

    def load(self, varname: str) -> ValueState:
        if varname not in self.varmap:
            op = Op(opname='var.incoming', bc_inst=None)
            vs = op.add_output(varname)
            self.incoming_vars[varname] = vs
            self.varmap[varname] = vs
        return self.varmap[varname]

    def replace_effect(self, env: ValueState):
        assert env.is_effect
        self.effect = env

    def convert(self, inst: dis.Instruction):
        fn = getattr(self, f'op_{inst.opname}')
        fn(inst)

    def set_kw_names(self, kw_vs: ValueState):
        assert self._kw_names is None
        self._kw_names = kw_vs

    def pop_kw_names(self):
        res = self._kw_names
        self._kw_names = None
        return res

    def op_POP_TOP(self, inst: dis.Instruction):
        self.pop()

    def op_RESUME(self, inst: dis.Instruction):
        pass

    def op_COPY_FREE_VARS(self, inst: dis.Instruction):
        pass

    def op_PUSH_NULL(self, inst: dis.Instruction):
        op = Op(opname='push_null', bc_inst=inst)
        null = op.add_output('null')
        self.push(null)

    def op_LOAD_GLOBAL(self, inst: dis.Instruction):
        assert isinstance(inst.arg, int)
        load_null = inst.arg & 1
        op = Op(opname='global', bc_inst=inst)
        op.add_input('env', self.effect)
        null = op.add_output('null')
        if load_null:
            self.push(null)
        self.push(op.add_output(f'{inst.argval}'))

    def op_LOAD_CONST(self, inst: dis.Instruction):
        op = Op(opname='const', bc_inst=inst)
        self.push(op.add_output('out'))

    def op_STORE_FAST(self, inst: dis.Instruction):
        tos = self.pop()
        op = Op(opname='store', bc_inst=inst)
        op.add_input('value', tos)
        varname = self._decorate_varname(inst.argval)
        self.store(varname, op.add_output(varname))

    def op_LOAD_FAST(self, inst: dis.Instruction):
        varname = self._decorate_varname(inst.argval)
        self.push(self.load(varname))

    def op_LOAD_ATTR(self, inst: dis.Instruction):
        obj = self.pop()
        attr = inst.argval
        op = Op(opname=f'load_attr.{attr}', bc_inst=inst)
        op.add_input('obj', obj)
        self.push(op.add_output('out'))

    def op_LOAD_METHOD(self, inst: dis.Instruction):
        obj = self.pop()
        attr = inst.argval
        op = Op(opname=f'load_method.{attr}', bc_inst=inst)
        op.add_input('obj', obj)
        self.push(op.add_output('null'))
        self.push(op.add_output('out'))

    def op_LOAD_DEREF(self, inst: dis.Instruction):
        op = Op(opname='load_deref', bc_inst=inst)
        self.push(op.add_output('out'))

    def op_PRECALL(self, inst: dis.Instruction):
        pass

    def op_KW_NAMES(self, inst: dis.Instruction):
        op = Op(opname='kw_names', bc_inst=inst)
        self.set_kw_names(op.add_output('out'))

    def op_CALL(self, inst: dis.Instruction):
        argc: int = inst.argval
        arg1plus = reversed([self.pop() for _ in range(argc)])
        arg0 = self.pop()
        kw_names = self.pop_kw_names()
        args: list[ValueState] = [arg0, *arg1plus]
        callable = self.pop()
        opname = 'call' if kw_names is None else 'call.kw'
        op = Op(opname=opname, bc_inst=inst)
        op.add_input('env', self.effect)
        op.add_input('callee', callable)
        for i, arg in enumerate(args):
            op.add_input(f'arg.{i}', arg)
        if kw_names is not None:
            op.add_input('kw_names', kw_names)
        self.replace_effect(op.add_output('env', is_effect=True))
        self.push(op.add_output('ret'))

    def op_GET_ITER(self, inst: dis.Instruction):
        tos = self.pop()
        op = Op(opname='getiter', bc_inst=inst)
        op.add_input('obj', tos)
        self.push(op.add_output('iter'))

    def op_FOR_ITER(self, inst: dis.Instruction):
        tos = self.top()
        op = Op(opname='foriter', bc_inst=inst)
        op.add_input('iter', tos)
        self.store('indvar', op.add_output('indvar'))

    def _binaryop(self, opname: str, inst: dis.Instruction):
        rhs = self.pop()
        lhs = self.pop()
        op = Op(opname=opname, bc_inst=inst)
        op.add_input('env', self.effect)
        op.add_input('lhs', lhs)
        op.add_input('rhs', rhs)
        self.replace_effect(op.add_output('env', is_effect=True))
        self.push(op.add_output('out'))

    def op_BINARY_OP(self, inst: dis.Instruction):
        self._binaryop('binaryop', inst)

    def op_COMPARE_OP(self, inst: dis.Instruction):
        self._binaryop('compareop', inst)

    def op_IS_OP(self, inst: dis.Instruction):
        self._binaryop('is_op', inst)

    def _unaryop(self, opname: str, inst: dis.Instruction):
        op = Op(opname=opname, bc_inst=inst)
        op.add_input('val', self.pop())
        self.push(op.add_output('out'))

    def op_UNARY_NOT(self, inst: dis.Instruction):
        self._unaryop('not', inst)

    def op_BINARY_SUBSCR(self, inst: dis.Instruction):
        index = self.pop()
        target = self.pop()
        op = Op(opname='binary_subscr', bc_inst=inst)
        op.add_input('env', self.effect)
        op.add_input('index', index)
        op.add_input('target', target)
        self.replace_effect(op.add_output('env', is_effect=True))
        self.push(op.add_output('out'))

    def op_STORE_SUBSCR(self, inst: dis.Instruction):
        index = self.pop()
        target = self.pop()
        value = self.pop()
        op = Op(opname='store_subscr', bc_inst=inst)
        op.add_input('env', self.effect)
        op.add_input('index', index)
        op.add_input('target', target)
        op.add_input('value', value)
        self.replace_effect(op.add_output('env', is_effect=True))

    def op_BUILD_TUPLE(self, inst: dis.Instruction):
        count = inst.arg
        assert isinstance(count, int)
        items = list(reversed([self.pop() for _ in range(count)]))
        op = Op(opname='build_tuple', bc_inst=inst)
        for i, it in enumerate(items):
            op.add_input(str(i), it)
        self.push(op.add_output('out'))

    def op_BUILD_SLICE(self, inst: dis.Instruction):
        argc = inst.arg
        if argc == 2:
            tos = self.pop()
            tos1 = self.pop()
            start = tos1
            stop = tos
            step = None
        elif argc == 3:
            tos = self.pop()
            tos1 = self.pop()
            tos2 = self.pop()
            start = tos2
            stop = tos1
            step = tos
        else:
            raise Exception('unreachable')
        op = Op(opname='build_slice', bc_inst=inst)
        op.add_input('start', start)
        op.add_input('stop', stop)
        if step is not None:
            op.add_input('step', step)
        self.push(op.add_output('out'))

    def op_RETURN_VALUE(self, inst: dis.Instruction):
        tos = self.pop()
        op = Op(opname='ret', bc_inst=inst)
        op.add_input('env', self.effect)
        op.add_input('retval', tos)
        self.replace_effect(op.add_output('env', is_effect=True))

    def op_RAISE_VARARGS(self, inst: dis.Instruction):
        if inst.arg == 0:
            exc = None
            raise NotImplementedError
        elif inst.arg == 1:
            exc = self.pop()
        else:
            raise ValueError('Multiple argument raise is not supported.')
        op = Op(opname='raise_varargs', bc_inst=inst)
        op.add_input('env', self.effect)
        op.add_input('exc', exc)
        self.replace_effect(op.add_output('env', is_effect=True))

    def op_JUMP_FORWARD(self, inst: dis.Instruction):
        pass

    def op_JUMP_BACKWARD(self, inst: dis.Instruction):
        pass

    def _POP_JUMP_X_IF_Y(self, inst: dis.Instruction, *, opname: str):
        tos = self.pop()
        op = Op(opname, bc_inst=inst)
        op.add_input('env', self.effect)
        op.add_input('pred', tos)
        self.replace_effect(op.add_output('env', is_effect=True))

    def op_POP_JUMP_FORWARD_IF_TRUE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname='jump.if_true')

    def op_POP_JUMP_FORWARD_IF_FALSE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname='jump.if_false')

    def op_POP_JUMP_BACKWARD_IF_TRUE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname='jump.if_true')

    def op_POP_JUMP_BACKWARD_IF_FALSE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname='jump.if_false')

    def op_POP_JUMP_FORWARD_IF_NONE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname='jump.if_none')

    def op_POP_JUMP_FORWARD_IF_NOT_NONE(self, inst: dis.Instruction):
        self._POP_JUMP_X_IF_Y(inst, opname='jump.if_not_none')

    def _JUMP_IF_X_OR_POP(self, inst: dis.Instruction, *, opname):
        tos = self.top()
        op = Op(opname, bc_inst=inst)
        op.add_input('env', self.effect)
        op.add_input('pred', tos)
        self.replace_effect(op.add_output('env', is_effect=True))

    def op_JUMP_IF_TRUE_OR_POP(self, inst: dis.Instruction):
        self._JUMP_IF_X_OR_POP(inst, opname='jump.if_true')

    def op_JUMP_IF_FALSE_OR_POP(self, inst: dis.Instruction):
        self._JUMP_IF_X_OR_POP(inst, opname='jump.if_false')