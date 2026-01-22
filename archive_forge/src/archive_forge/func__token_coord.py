import warnings
def _token_coord(self, p, token_idx):
    """ Returns the coordinates for the YaccProduction object 'p' indexed
            with 'token_idx'. The coordinate includes the 'lineno' and
            'column'. Both follow the lex semantic, starting from 1.
        """
    last_cr = p.lexer.lexer.lexdata.rfind('\n', 0, p.lexpos(token_idx))
    if last_cr < 0:
        last_cr = -1
    column = p.lexpos(token_idx) - last_cr
    return self._coord(p.lineno(token_idx), column)