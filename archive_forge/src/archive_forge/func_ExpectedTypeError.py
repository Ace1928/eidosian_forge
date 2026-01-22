def ExpectedTypeError(value, expected, param):
    """error message when param was supposed to be one type, but found another"""
    name = type_name(value)
    return TypeError('%s must be %s, not %s' % (param, expected, name))