import string as string_module
from yaql.language import specs
from yaql.language import utils
from yaql.language import yaqltypes
@specs.parameter('string', yaqltypes.String())
@specs.method
def to_upper(string):
    """:yaql:toUpper

    Returns a string with all case-based characters uppercase.

    :signature: string.toUpper()
    :receiverArg string: value to uppercase
    :argType string: string
    :returnType: string

    .. code::

        yaql> "aB1c".toUpper()
        "AB1C"
    """
    return string.upper()