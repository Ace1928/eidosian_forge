import ast
import io
import sys
import tokenize
def _fstring_FormattedValue(self, t, write):
    write('{')
    expr = io.StringIO()
    Unparser(t.value, expr)
    expr = expr.getvalue().rstrip('\n')
    if expr.startswith('{'):
        write(' ')
    write(expr)
    if t.conversion != -1:
        conversion = chr(t.conversion)
        assert conversion in 'sra'
        write(f'!{conversion}')
    if t.format_spec:
        write(':')
        meth = getattr(self, '_fstring_' + type(t.format_spec).__name__)
        meth(t.format_spec, write)
    write('}')