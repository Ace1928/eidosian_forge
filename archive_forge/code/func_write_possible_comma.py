from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def write_possible_comma():
    if _first[0]:
        _first[0] = False
    else:
        self._write(', ')