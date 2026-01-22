def ExpectedStringError(value, param):
    """error message when param was supposed to be unicode or bytes"""
    return ExpectedTypeError(value, 'unicode or bytes', param)