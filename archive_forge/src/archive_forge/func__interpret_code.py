from __future__ import absolute_import
import re
import sys
import os
import tokenize
from io import StringIO
from ._looper import looper
from .compat3 import bytes, unicode_, basestring_, next, is_unicode, coerce_text
def _interpret_code(self, code, ns, out, defs):
    __traceback_hide__ = True
    name, pos = (code[0], code[1])
    if name == 'py':
        self._exec(code[2], ns, pos)
    elif name == 'continue':
        raise _TemplateContinue()
    elif name == 'break':
        raise _TemplateBreak()
    elif name == 'for':
        vars, expr, content = (code[2], code[3], code[4])
        expr = self._eval(expr, ns, pos)
        self._interpret_for(vars, expr, content, ns, out, defs)
    elif name == 'cond':
        parts = code[2:]
        self._interpret_if(parts, ns, out, defs)
    elif name == 'expr':
        parts = code[2].split('|')
        base = self._eval(parts[0], ns, pos)
        for part in parts[1:]:
            func = self._eval(part, ns, pos)
            base = func(base)
        out.append(self._repr(base, pos))
    elif name == 'default':
        var, expr = (code[2], code[3])
        if var not in ns:
            result = self._eval(expr, ns, pos)
            ns[var] = result
    elif name == 'inherit':
        expr = code[2]
        value = self._eval(expr, ns, pos)
        defs['__inherit__'] = value
    elif name == 'def':
        name = code[2]
        signature = code[3]
        parts = code[4]
        ns[name] = defs[name] = TemplateDef(self, name, signature, body=parts, ns=ns, pos=pos)
    elif name == 'comment':
        return
    else:
        assert 0, 'Unknown code: %r' % name