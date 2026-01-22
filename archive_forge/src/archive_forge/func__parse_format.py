import re
import numpy as np
def _parse_format(self, tokens):
    if not tokens[0].type == 'LPAR':
        raise SyntaxError("Expected left parenthesis at position %d (got '%s')" % (0, tokens[0].value))
    elif not tokens[-1].type == 'RPAR':
        raise SyntaxError("Expected right parenthesis at position %d (got '%s')" % (len(tokens), tokens[-1].value))
    tokens = tokens[1:-1]
    types = [t.type for t in tokens]
    if types[0] == 'INT':
        repeat = int(tokens.pop(0).value)
    else:
        repeat = None
    next = tokens.pop(0)
    if next.type == 'INT_ID':
        next = self._next(tokens, 'INT')
        width = int(next.value)
        if tokens:
            min = int(self._get_min(tokens))
        else:
            min = None
        return IntFormat(width, min, repeat)
    elif next.type == 'EXP_ID':
        next = self._next(tokens, 'INT')
        width = int(next.value)
        next = self._next(tokens, 'DOT')
        next = self._next(tokens, 'INT')
        significand = int(next.value)
        if tokens:
            next = self._next(tokens, 'EXP_ID')
            next = self._next(tokens, 'INT')
            min = int(next.value)
        else:
            min = None
        return ExpFormat(width, significand, min, repeat)
    else:
        raise SyntaxError('Invalid formatter type %s' % next.value)