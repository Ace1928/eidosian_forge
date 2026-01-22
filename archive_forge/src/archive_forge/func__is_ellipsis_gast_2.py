import functools
import gast
def _is_ellipsis_gast_2(node):
    return isinstance(node, gast.Ellipsis)