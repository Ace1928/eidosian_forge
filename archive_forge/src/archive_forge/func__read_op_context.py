from __future__ import print_function
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def _read_op_context(token, c):
    if token.type == Token.RPAREN:
        if c.trace:
            print('Found close-paren')
        while c.op_stack and c.op_stack[-1].op.token_type != Token.LPAREN:
            _run_op(c)
        if not c.op_stack:
            raise PatsyError("missing '(' or extra ')'", token)
        assert c.op_stack[-1].op.token_type == Token.LPAREN
        combined = Origin.combine([c.op_stack[-1].token, c.noun_stack[-1].token, token])
        c.noun_stack[-1].origin = combined
        c.op_stack.pop()
        return False
    elif token.type in c.binary_ops:
        if c.trace:
            print('Found binary operator %r' % token.type)
        stackop = _StackOperator(c.binary_ops[token.type], token)
        while c.op_stack and stackop.op.precedence <= c.op_stack[-1].op.precedence:
            _run_op(c)
        if c.trace:
            print('Pushing binary operator %r' % token.type)
        c.op_stack.append(stackop)
        return True
    else:
        raise PatsyError("expected an operator, not '%s'" % (token.origin.relevant_code(),), token)