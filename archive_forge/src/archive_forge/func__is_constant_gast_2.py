import functools
import gast
def _is_constant_gast_2(node):
    return isinstance(node, (gast.Num, gast.Str, gast.Bytes, gast.Ellipsis, gast.NameConstant))