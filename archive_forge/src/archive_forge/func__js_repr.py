from ..utils import SchemaBase
def _js_repr(val):
    """Return a javascript-safe string representation of val"""
    if val is True:
        return 'true'
    elif val is False:
        return 'false'
    elif val is None:
        return 'null'
    elif isinstance(val, OperatorMixin):
        return val._to_expr()
    else:
        return repr(val)