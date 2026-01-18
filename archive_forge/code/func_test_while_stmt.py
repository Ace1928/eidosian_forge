from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises, XFAIL
from sympy.external import import_module
def test_while_stmt():
    c_src1 = 'void func()' + '{' + '\n' + 'int i = 0;' + '\n' + 'while(i < 10)' + '\n' + '{' + '\n' + 'i++;' + '\n' + '}}'
    c_src2 = 'void func()' + '{' + '\n' + 'int i = 0;' + '\n' + 'while(i < 10)' + '\n' + 'i++;' + '\n' + '}'
    c_src3 = 'void func()' + '{' + '\n' + 'int i = 10;' + '\n' + 'int cnt = 0;' + '\n' + 'while(i > 0)' + '\n' + '{' + '\n' + 'i--;' + '\n' + 'cnt++;' + '\n' + '}' + '\n' + '}'
    c_src4 = 'int digit_sum(int n)' + '{' + '\n' + 'int sum = 0;' + '\n' + 'while(n > 0)' + '\n' + '{' + '\n' + 'sum += (n % 10);' + '\n' + 'n /= 10;' + '\n' + '}' + '\n' + 'return sum;' + '\n' + '}'
    c_src5 = 'void func()' + '{' + '\n' + 'while(1);' + '\n' + '}'
    res1 = SymPyExpression(c_src1, 'c').return_expr()
    res2 = SymPyExpression(c_src2, 'c').return_expr()
    res3 = SymPyExpression(c_src3, 'c').return_expr()
    res4 = SymPyExpression(c_src4, 'c').return_expr()
    res5 = SymPyExpression(c_src5, 'c').return_expr()
    assert res1[0] == FunctionDefinition(NoneToken(), name=String('func'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('i'), type=IntBaseType(String('intc')), value=Integer(0))), While(StrictLessThan(Symbol('i'), Integer(10)), body=CodeBlock(PostIncrement(Symbol('i'))))))
    assert res2[0] == res1[0]
    assert res3[0] == FunctionDefinition(NoneToken(), name=String('func'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('i'), type=IntBaseType(String('intc')), value=Integer(10))), Declaration(Variable(Symbol('cnt'), type=IntBaseType(String('intc')), value=Integer(0))), While(StrictGreaterThan(Symbol('i'), Integer(0)), body=CodeBlock(PostDecrement(Symbol('i')), PostIncrement(Symbol('cnt'))))))
    assert res4[0] == FunctionDefinition(IntBaseType(String('intc')), name=String('digit_sum'), parameters=(Variable(Symbol('n'), type=IntBaseType(String('intc'))),), body=CodeBlock(Declaration(Variable(Symbol('sum'), type=IntBaseType(String('intc')), value=Integer(0))), While(StrictGreaterThan(Symbol('n'), Integer(0)), body=CodeBlock(AddAugmentedAssignment(Variable(Symbol('sum')), Mod(Symbol('n'), Integer(10))), DivAugmentedAssignment(Variable(Symbol('n')), Integer(10)))), Return('sum')))
    assert res5[0] == FunctionDefinition(NoneToken(), name=String('func'), parameters=(), body=CodeBlock(While(Integer(1), body=CodeBlock(NoneToken()))))