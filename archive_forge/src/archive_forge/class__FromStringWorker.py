import re
import warnings
from enum import Enum
from math import gcd
class _FromStringWorker:

    def __init__(self, language=Language.C):
        self.original = None
        self.quotes_map = None
        self.language = language

    def finalize_string(self, s):
        return insert_quotes(s, self.quotes_map)

    def parse(self, inp):
        self.original = inp
        unquoted, self.quotes_map = eliminate_quotes(inp)
        return self.process(unquoted)

    def process(self, s, context='expr'):
        """Parse string within the given context.

        The context may define the result in case of ambiguous
        expressions. For instance, consider expressions `f(x, y)` and
        `(x, y) + (a, b)` where `f` is a function and pair `(x, y)`
        denotes complex number. Specifying context as "args" or
        "expr", the subexpression `(x, y)` will be parse to an
        argument list or to a complex number, respectively.
        """
        if isinstance(s, (list, tuple)):
            return type(s)((self.process(s_, context) for s_ in s))
        assert isinstance(s, str), (type(s), s)
        r, raw_symbols_map = replace_parenthesis(s)
        r = r.strip()

        def restore(r):
            if isinstance(r, (list, tuple)):
                return type(r)(map(restore, r))
            return unreplace_parenthesis(r, raw_symbols_map)
        if ',' in r:
            operands = restore(r.split(','))
            if context == 'args':
                return tuple(self.process(operands))
            if context == 'expr':
                if len(operands) == 2:
                    return as_complex(*self.process(operands))
            raise NotImplementedError(f'parsing comma-separated list (context={context}): {r}')
        m = re.match('\\A([^?]+)[?]([^:]+)[:](.+)\\Z', r)
        if m:
            assert context == 'expr', context
            oper, expr1, expr2 = restore(m.groups())
            oper = self.process(oper)
            expr1 = self.process(expr1)
            expr2 = self.process(expr2)
            return as_ternary(oper, expr1, expr2)
        if self.language is Language.Fortran:
            m = re.match('\\A(.+)\\s*[.](eq|ne|lt|le|gt|ge)[.]\\s*(.+)\\Z', r, re.I)
        else:
            m = re.match('\\A(.+)\\s*([=][=]|[!][=]|[<][=]|[<]|[>][=]|[>])\\s*(.+)\\Z', r)
        if m:
            left, rop, right = m.groups()
            if self.language is Language.Fortran:
                rop = '.' + rop + '.'
            left, right = self.process(restore((left, right)))
            rop = RelOp.fromstring(rop, language=self.language)
            return Expr(Op.RELATIONAL, (rop, left, right))
        m = re.match('\\A(\\w[\\w\\d_]*)\\s*[=](.*)\\Z', r)
        if m:
            keyname, value = m.groups()
            value = restore(value)
            return _Pair(keyname, self.process(value))
        operands = re.split('((?<!\\d[edED])[+-])', r)
        if len(operands) > 1:
            result = self.process(restore(operands[0] or '0'))
            for op, operand in zip(operands[1::2], operands[2::2]):
                operand = self.process(restore(operand))
                op = op.strip()
                if op == '+':
                    result += operand
                else:
                    assert op == '-'
                    result -= operand
            return result
        if self.language is Language.Fortran and '//' in r:
            operands = restore(r.split('//'))
            return Expr(Op.CONCAT, tuple(self.process(operands)))
        operands = re.split('(?<=[@\\w\\d_])\\s*([*]|/)', r if self.language is Language.C else r.replace('**', '@__f2py_DOUBLE_STAR@'))
        if len(operands) > 1:
            operands = restore(operands)
            if self.language is not Language.C:
                operands = [operand.replace('@__f2py_DOUBLE_STAR@', '**') for operand in operands]
            result = self.process(operands[0])
            for op, operand in zip(operands[1::2], operands[2::2]):
                operand = self.process(operand)
                op = op.strip()
                if op == '*':
                    result *= operand
                else:
                    assert op == '/'
                    result /= operand
            return result
        if r.startswith('*') or r.startswith('&'):
            op = {'*': Op.DEREF, '&': Op.REF}[r[0]]
            operand = self.process(restore(r[1:]))
            return Expr(op, operand)
        if self.language is not Language.C and '**' in r:
            operands = list(reversed(restore(r.split('**'))))
            result = self.process(operands[0])
            for operand in operands[1:]:
                operand = self.process(operand)
                result = operand ** result
            return result
        m = re.match('\\A({digit_string})({kind}|)\\Z'.format(digit_string='\\d+', kind='_(\\d+|\\w[\\w\\d_]*)'), r)
        if m:
            value, _, kind = m.groups()
            if kind and kind.isdigit():
                kind = int(kind)
            return as_integer(int(value), kind or 4)
        m = re.match('\\A({significant}({exponent}|)|\\d+{exponent})({kind}|)\\Z'.format(significant='[.]\\d+|\\d+[.]\\d*', exponent='[edED][+-]?\\d+', kind='_(\\d+|\\w[\\w\\d_]*)'), r)
        if m:
            value, _, _, kind = m.groups()
            if kind and kind.isdigit():
                kind = int(kind)
            value = value.lower()
            if 'd' in value:
                return as_real(float(value.replace('d', 'e')), kind or 8)
            return as_real(float(value), kind or 4)
        if r in self.quotes_map:
            kind = r[:r.find('@')]
            return as_string(self.quotes_map[r], kind or 1)
        if r in raw_symbols_map:
            paren = _get_parenthesis_kind(r)
            items = self.process(restore(raw_symbols_map[r]), 'expr' if paren == 'ROUND' else 'args')
            if paren == 'ROUND':
                if isinstance(items, Expr):
                    return items
            if paren in ['ROUNDDIV', 'SQUARE']:
                if isinstance(items, Expr):
                    items = (items,)
                return as_array(items)
        m = re.match('\\A(.+)\\s*(@__f2py_PARENTHESIS_(ROUND|SQUARE)_\\d+@)\\Z', r)
        if m:
            target, args, paren = m.groups()
            target = self.process(restore(target))
            args = self.process(restore(args)[1:-1], 'args')
            if not isinstance(args, tuple):
                args = (args,)
            if paren == 'ROUND':
                kwargs = dict(((a.left, a.right) for a in args if isinstance(a, _Pair)))
                args = tuple((a for a in args if not isinstance(a, _Pair)))
                return as_apply(target, *args, **kwargs)
            else:
                assert paren == 'SQUARE'
                return target[args]
        m = re.match('\\A\\w[\\w\\d_]*\\Z', r)
        if m:
            return as_symbol(r)
        r = self.finalize_string(restore(r))
        ewarn(f'fromstring: treating {r!r} as symbol (original={self.original})')
        return as_symbol(r)