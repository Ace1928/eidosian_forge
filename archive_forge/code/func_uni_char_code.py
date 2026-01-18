import json
from ..error import GraphQLSyntaxError
def uni_char_code(a, b, c, d):
    """Converts four hexidecimal chars to the integer that the
    string represents. For example, uniCharCode('0','0','0','f')
    will return 15, and uniCharCode('0','0','f','f') returns 255.

    Returns a negative number on error, if a char was invalid.

    This is implemented by noting that char2hex() returns -1 on error,
    which means the result of ORing the char2hex() will also be negative.
    """
    return char2hex(a) << 12 | char2hex(b) << 8 | char2hex(c) << 4 | char2hex(d)