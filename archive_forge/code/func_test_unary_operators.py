from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises, XFAIL
from sympy.external import import_module
def test_unary_operators():
    c_src1 = 'void func()' + '{' + '\n' + 'int a = 10;' + '\n' + 'int b = 20;' + '\n' + '++a;' + '\n' + '--b;' + '\n' + 'a++;' + '\n' + 'b--;' + '\n' + '}'
    c_src2 = 'void func()' + '{' + '\n' + 'int a = 10;' + '\n' + 'int b = -100;' + '\n' + 'int c = +19;' + '\n' + 'int d = ++a;' + '\n' + 'int e = --b;' + '\n' + 'int f = a++;' + '\n' + 'int g = b--;' + '\n' + 'bool h = !false;' + '\n' + 'bool i = !d;' + '\n' + 'bool j = !0;' + '\n' + 'bool k = !10.0;' + '\n' + '}'
    c_src_raise1 = 'void func()' + '{' + '\n' + 'int a = 10;' + '\n' + 'int b = ~a;' + '\n' + '}'
    c_src_raise2 = 'void func()' + '{' + '\n' + 'int a = 10;' + '\n' + 'int b = *&a;' + '\n' + '}'
    res1 = SymPyExpression(c_src1, 'c').return_expr()
    res2 = SymPyExpression(c_src2, 'c').return_expr()
    assert res1[0] == FunctionDefinition(NoneToken(), name=String('func'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(10))), Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(20))), PreIncrement(Symbol('a')), PreDecrement(Symbol('b')), PostIncrement(Symbol('a')), PostDecrement(Symbol('b'))))
    assert res2[0] == FunctionDefinition(NoneToken(), name=String('func'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(10))), Declaration(Variable(Symbol('b'), type=IntBaseType(String('intc')), value=Integer(-100))), Declaration(Variable(Symbol('c'), type=IntBaseType(String('intc')), value=Integer(19))), Declaration(Variable(Symbol('d'), type=IntBaseType(String('intc')), value=PreIncrement(Symbol('a')))), Declaration(Variable(Symbol('e'), type=IntBaseType(String('intc')), value=PreDecrement(Symbol('b')))), Declaration(Variable(Symbol('f'), type=IntBaseType(String('intc')), value=PostIncrement(Symbol('a')))), Declaration(Variable(Symbol('g'), type=IntBaseType(String('intc')), value=PostDecrement(Symbol('b')))), Declaration(Variable(Symbol('h'), type=Type(String('bool')), value=true)), Declaration(Variable(Symbol('i'), type=Type(String('bool')), value=Not(Symbol('d')))), Declaration(Variable(Symbol('j'), type=Type(String('bool')), value=true)), Declaration(Variable(Symbol('k'), type=Type(String('bool')), value=false))))
    raises(NotImplementedError, lambda: SymPyExpression(c_src_raise1, 'c'))
    raises(NotImplementedError, lambda: SymPyExpression(c_src_raise2, 'c'))