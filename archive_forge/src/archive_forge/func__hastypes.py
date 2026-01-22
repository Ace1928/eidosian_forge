from sympy.core import Basic
def _hastypes(self, expr, types):
    """Check if ``expr`` is any of ``types``. """
    _types = [cls.__name__ for cls in expr.__class__.mro()]
    return bool(set(_types).intersection(types))