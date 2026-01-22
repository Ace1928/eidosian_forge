from . import c_ast
def _parenthesize_unless_simple(self, n):
    """ Common use case for _parenthesize_if
        """
    return self._parenthesize_if(n, lambda d: not self._is_simple_node(d))