import re
import operator
from fractions import Fraction
import sys
def _parse_polynomial_from_string(s, parse_coefficient_function):
    operand_stack = []
    operator_stack = []
    no_operand_since_opening_parenthesis = [True]

    def debug_print(s):
        print('=' * 75)
        print('Remaining string : ', s)
        print('Operator Stack   : ', operator_stack)
        print('Operand Stack    : ', operand_stack)

    def eval_preceding_operators_on_stack(operator=None):
        while operator_stack:
            top_operator = operator_stack[-1]
            if top_operator == '(':
                return
            if _operator_precedence[top_operator] < _operator_precedence[operator]:
                return
            top_operator = operator_stack.pop()
            r = operand_stack.pop()
            l = operand_stack.pop()
            operand_stack.append(_apply_operator(top_operator, l, r))

    def process_next_token(s):
        s = s.lstrip()
        constant, rest = parse_coefficient_function(s)
        if constant is not None:
            operand_stack.append(Polynomial.constant_polynomial(constant))
            no_operand_since_opening_parenthesis[0] = False
            return rest
        variable, rest = _parse_variable(s)
        if variable:
            operand_stack.append(Polynomial.from_variable_name(variable))
            no_operand_since_opening_parenthesis[0] = False
            return rest
        next_char, rest = (s[0], s[1:])
        if next_char in list(_operators.keys()):
            operator = next_char
            eval_preceding_operators_on_stack(operator)
            operator_stack.append(operator)
            if operator in '+-':
                if no_operand_since_opening_parenthesis[0]:
                    operand_stack.append(Polynomial())
                    no_operand_since_opening_parenthesis[0] = False
            return rest
        if next_char in '()':
            parenthesis = next_char
            if parenthesis == '(':
                operator_stack.append('(')
                no_operand_since_opening_parenthesis[0] = True
            else:
                eval_preceding_operators_on_stack()
                top_operator = operator_stack.pop()
                assert top_operator == '('
            return rest
        raise Exception('While parsing polynomial %s' % s)
    s = s.strip()
    while s:
        s = process_next_token(s)
    eval_preceding_operators_on_stack(None)
    assert not operator_stack
    if not operand_stack:
        return Polynomial(())
    assert len(operand_stack) == 1 or (len(operand_stack) == 2 and operand_stack[0] == Polynomial())
    return operand_stack[-1]