from genshi.compat import ast as _ast, _ast_Constant, IS_PYTHON2, isstring, \
def _clone(self, node):
    clone = node.__class__()
    for name in getattr(clone, '_attributes', ()):
        try:
            setattr(clone, name, getattr(node, name))
        except AttributeError:
            pass
    for name in clone._fields:
        try:
            value = getattr(node, name)
        except AttributeError:
            pass
        else:
            if value is None:
                pass
            elif isinstance(value, list):
                value = [self.visit(x) for x in value]
            elif isinstance(value, tuple):
                value = tuple((self.visit(x) for x in value))
            else:
                value = self.visit(value)
            setattr(clone, name, value)
    return clone