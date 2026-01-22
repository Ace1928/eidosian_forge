import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import sys
import textwrap
import types
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.concrete import OFFSET_AS_INSTRUCTION
from _pydevd_frame_eval.vendored.bytecode.tests import get_code, TestCase
class ConcreteBytecodeTests(TestCase):

    def test_repr(self):
        r = repr(ConcreteBytecode())
        self.assertIn('ConcreteBytecode', r)
        self.assertIn('0', r)

    def test_eq(self):
        code = ConcreteBytecode()
        self.assertFalse(code == 1)
        for name, val in (('names', ['a']), ('varnames', ['a']), ('consts', [1]), ('argcount', 1), ('kwonlyargcount', 2), ('flags', CompilerFlags(CompilerFlags.GENERATOR)), ('first_lineno', 10), ('filename', 'xxxx.py'), ('name', '__x'), ('docstring', 'x-x-x'), ('cellvars', [CellVar('x')]), ('freevars', [FreeVar('x')])):
            c = ConcreteBytecode()
            setattr(c, name, val)
            self.assertFalse(code == c)
        if sys.version_info > (3, 8):
            c = ConcreteBytecode()
            c.posonlyargcount = 10
            self.assertFalse(code == c)
        c = ConcreteBytecode()
        c.consts = [1]
        code.consts = [1]
        c.append(ConcreteInstr('LOAD_CONST', 0))
        self.assertFalse(code == c)

    def test_attr(self):
        code_obj = get_code('x = 5')
        code = ConcreteBytecode.from_code(code_obj)
        self.assertEqual(code.consts, [5, None])
        self.assertEqual(code.names, ['x'])
        self.assertEqual(code.varnames, [])
        self.assertEqual(code.freevars, [])
        self.assertListEqual(list(code), [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('STORE_NAME', 0, lineno=1), ConcreteInstr('LOAD_CONST', 1, lineno=1), ConcreteInstr('RETURN_VALUE', lineno=1)])

    def test_invalid_types(self):
        code = ConcreteBytecode()
        code.append(Label())
        with self.assertRaises(ValueError):
            list(code)
        with self.assertRaises(ValueError):
            code.legalize()
        with self.assertRaises(ValueError):
            ConcreteBytecode([Label()])

    def test_to_code_lnotab(self):

        def f():
            x = 7
            y = 8
            z = 9
        fl = f.__code__.co_firstlineno
        concrete = ConcreteBytecode()
        concrete.consts = [None, 7, 8, 9]
        concrete.varnames = ['x', 'y', 'z']
        concrete.first_lineno = fl
        concrete.extend([SetLineno(fl + 3), ConcreteInstr('LOAD_CONST', 1), ConcreteInstr('STORE_FAST', 0), SetLineno(fl + 4), ConcreteInstr('LOAD_CONST', 2), ConcreteInstr('STORE_FAST', 1), SetLineno(fl + 5), ConcreteInstr('LOAD_CONST', 3), ConcreteInstr('STORE_FAST', 2), ConcreteInstr('LOAD_CONST', 0), ConcreteInstr('RETURN_VALUE')])
        code = concrete.to_code()
        self.assertEqual(code.co_code, f.__code__.co_code)
        self.assertEqual(code.co_lnotab, f.__code__.co_lnotab)
        if sys.version_info >= (3, 10):
            self.assertEqual(code.co_linetable, f.__code__.co_linetable)

    def test_negative_lnotab(self):
        concrete = ConcreteBytecode([ConcreteInstr('LOAD_CONST', 0), ConcreteInstr('STORE_NAME', 0), SetLineno(2), ConcreteInstr('LOAD_CONST', 1), ConcreteInstr('STORE_NAME', 1)])
        concrete.consts = [7, 8]
        concrete.names = ['x', 'y']
        concrete.first_lineno = 5
        code = concrete.to_code()
        expected = b'd\x00Z\x00d\x01Z\x01'
        self.assertEqual(code.co_code, expected)
        self.assertEqual(code.co_firstlineno, 5)
        self.assertEqual(code.co_lnotab, b'\x04\xfd')

    def test_extended_lnotab(self):
        concrete = ConcreteBytecode([ConcreteInstr('LOAD_CONST', 0), SetLineno(1 + 128), ConcreteInstr('STORE_NAME', 0), SetLineno(1 + 129), ConcreteInstr('LOAD_CONST', 1), SetLineno(1), ConcreteInstr('STORE_NAME', 1)])
        concrete.consts = [7, 8]
        concrete.names = ['x', 'y']
        concrete.first_lineno = 1
        code = concrete.to_code()
        expected = b'd\x00Z\x00d\x01Z\x01'
        self.assertEqual(code.co_code, expected)
        self.assertEqual(code.co_firstlineno, 1)
        self.assertEqual(code.co_lnotab, b'\x02\x7f\x00\x01\x02\x01\x02\x80\x00\xff')

    def test_extended_lnotab2(self):
        base_code = compile('x = 7' + '\n' * 200 + 'y = 8', '', 'exec')
        concrete = ConcreteBytecode([ConcreteInstr('LOAD_CONST', 0), ConcreteInstr('STORE_NAME', 0), SetLineno(201), ConcreteInstr('LOAD_CONST', 1), ConcreteInstr('STORE_NAME', 1), ConcreteInstr('LOAD_CONST', 2), ConcreteInstr('RETURN_VALUE')])
        concrete.consts = [None, 7, 8]
        concrete.names = ['x', 'y']
        concrete.first_lineno = 1
        code = concrete.to_code()
        self.assertEqual(code.co_code, base_code.co_code)
        self.assertEqual(code.co_firstlineno, base_code.co_firstlineno)
        self.assertEqual(code.co_lnotab, base_code.co_lnotab)
        if sys.version_info >= (3, 10):
            self.assertEqual(code.co_linetable, base_code.co_linetable)

    def test_to_bytecode_consts(self):
        code = ConcreteBytecode()
        code.consts = [0.0, None, -0.0, 0.0]
        code.names = ['x', 'y']
        code.extend([ConcreteInstr('LOAD_CONST', 2, lineno=1), ConcreteInstr('STORE_NAME', 0, lineno=1), ConcreteInstr('LOAD_CONST', 3, lineno=2), ConcreteInstr('STORE_NAME', 1, lineno=2), ConcreteInstr('LOAD_CONST', 1, lineno=2), ConcreteInstr('RETURN_VALUE', lineno=2)])
        code = code.to_bytecode().to_concrete_bytecode()
        self.assertEqual(code.consts, [-0.0, 0.0, None])
        code.names = ['x', 'y']
        self.assertListEqual(list(code), [ConcreteInstr('LOAD_CONST', 0, lineno=1), ConcreteInstr('STORE_NAME', 0, lineno=1), ConcreteInstr('LOAD_CONST', 1, lineno=2), ConcreteInstr('STORE_NAME', 1, lineno=2), ConcreteInstr('LOAD_CONST', 2, lineno=2), ConcreteInstr('RETURN_VALUE', lineno=2)])

    def test_cellvar(self):
        concrete = ConcreteBytecode()
        concrete.cellvars = ['x']
        concrete.append(ConcreteInstr('LOAD_DEREF', 0))
        code = concrete.to_code()
        concrete = ConcreteBytecode.from_code(code)
        self.assertEqual(concrete.cellvars, ['x'])
        self.assertEqual(concrete.freevars, [])
        self.assertEqual(list(concrete), [ConcreteInstr('LOAD_DEREF', 0, lineno=1)])
        bytecode = concrete.to_bytecode()
        self.assertEqual(bytecode.cellvars, ['x'])
        self.assertEqual(list(bytecode), [Instr('LOAD_DEREF', CellVar('x'), lineno=1)])

    def test_freevar(self):
        concrete = ConcreteBytecode()
        concrete.freevars = ['x']
        concrete.append(ConcreteInstr('LOAD_DEREF', 0))
        code = concrete.to_code()
        concrete = ConcreteBytecode.from_code(code)
        self.assertEqual(concrete.cellvars, [])
        self.assertEqual(concrete.freevars, ['x'])
        self.assertEqual(list(concrete), [ConcreteInstr('LOAD_DEREF', 0, lineno=1)])
        bytecode = concrete.to_bytecode()
        self.assertEqual(bytecode.cellvars, [])
        self.assertEqual(list(bytecode), [Instr('LOAD_DEREF', FreeVar('x'), lineno=1)])

    def test_cellvar_freevar(self):
        concrete = ConcreteBytecode()
        concrete.cellvars = ['cell']
        concrete.freevars = ['free']
        concrete.append(ConcreteInstr('LOAD_DEREF', 0))
        concrete.append(ConcreteInstr('LOAD_DEREF', 1))
        code = concrete.to_code()
        concrete = ConcreteBytecode.from_code(code)
        self.assertEqual(concrete.cellvars, ['cell'])
        self.assertEqual(concrete.freevars, ['free'])
        self.assertEqual(list(concrete), [ConcreteInstr('LOAD_DEREF', 0, lineno=1), ConcreteInstr('LOAD_DEREF', 1, lineno=1)])
        bytecode = concrete.to_bytecode()
        self.assertEqual(bytecode.cellvars, ['cell'])
        self.assertEqual(list(bytecode), [Instr('LOAD_DEREF', CellVar('cell'), lineno=1), Instr('LOAD_DEREF', FreeVar('free'), lineno=1)])

    def test_load_classderef(self):
        concrete = ConcreteBytecode()
        concrete.cellvars = ['__class__']
        concrete.freevars = ['__class__']
        concrete.extend([ConcreteInstr('LOAD_CLASSDEREF', 1), ConcreteInstr('STORE_DEREF', 1)])
        bytecode = concrete.to_bytecode()
        self.assertEqual(bytecode.freevars, ['__class__'])
        self.assertEqual(bytecode.cellvars, ['__class__'])
        self.assertEqual(list(bytecode), [Instr('LOAD_CLASSDEREF', FreeVar('__class__'), lineno=1), Instr('STORE_DEREF', FreeVar('__class__'), lineno=1)])
        concrete = bytecode.to_concrete_bytecode()
        self.assertEqual(concrete.freevars, ['__class__'])
        self.assertEqual(concrete.cellvars, ['__class__'])
        self.assertEqual(list(concrete), [ConcreteInstr('LOAD_CLASSDEREF', 1, lineno=1), ConcreteInstr('STORE_DEREF', 1, lineno=1)])
        code = concrete.to_code()
        self.assertEqual(code.co_freevars, ('__class__',))
        self.assertEqual(code.co_cellvars, ('__class__',))
        self.assertEqual(code.co_code, b'\x94\x01\x89\x01')

    def test_explicit_stacksize(self):
        code_obj = get_code("print('%s' % (a,b,c))")
        original_stacksize = code_obj.co_stacksize
        concrete = ConcreteBytecode.from_code(code_obj)
        explicit_stacksize = original_stacksize + 42
        new_code_obj = concrete.to_code(stacksize=explicit_stacksize)
        self.assertEqual(new_code_obj.co_stacksize, explicit_stacksize)
        explicit_stacksize = 0
        new_code_obj = concrete.to_code(stacksize=explicit_stacksize)
        self.assertEqual(new_code_obj.co_stacksize, explicit_stacksize)

    def test_legalize(self):
        concrete = ConcreteBytecode()
        concrete.first_lineno = 3
        concrete.consts = [7, 8, 9]
        concrete.names = ['x', 'y', 'z']
        concrete.extend([ConcreteInstr('LOAD_CONST', 0), ConcreteInstr('STORE_NAME', 0), ConcreteInstr('LOAD_CONST', 1, lineno=4), ConcreteInstr('STORE_NAME', 1), SetLineno(5), ConcreteInstr('LOAD_CONST', 2, lineno=6), ConcreteInstr('STORE_NAME', 2)])
        concrete.legalize()
        self.assertListEqual(list(concrete), [ConcreteInstr('LOAD_CONST', 0, lineno=3), ConcreteInstr('STORE_NAME', 0, lineno=3), ConcreteInstr('LOAD_CONST', 1, lineno=4), ConcreteInstr('STORE_NAME', 1, lineno=4), ConcreteInstr('LOAD_CONST', 2, lineno=5), ConcreteInstr('STORE_NAME', 2, lineno=5)])

    def test_slice(self):
        concrete = ConcreteBytecode()
        concrete.first_lineno = 3
        concrete.consts = [7, 8, 9]
        concrete.names = ['x', 'y', 'z']
        concrete.extend([ConcreteInstr('LOAD_CONST', 0), ConcreteInstr('STORE_NAME', 0), SetLineno(4), ConcreteInstr('LOAD_CONST', 1), ConcreteInstr('STORE_NAME', 1), SetLineno(5), ConcreteInstr('LOAD_CONST', 2), ConcreteInstr('STORE_NAME', 2)])
        self.assertEqual(concrete, concrete[:])

    def test_copy(self):
        concrete = ConcreteBytecode()
        concrete.first_lineno = 3
        concrete.consts = [7, 8, 9]
        concrete.names = ['x', 'y', 'z']
        concrete.extend([ConcreteInstr('LOAD_CONST', 0), ConcreteInstr('STORE_NAME', 0), SetLineno(4), ConcreteInstr('LOAD_CONST', 1), ConcreteInstr('STORE_NAME', 1), SetLineno(5), ConcreteInstr('LOAD_CONST', 2), ConcreteInstr('STORE_NAME', 2)])
        self.assertEqual(concrete, concrete.copy())