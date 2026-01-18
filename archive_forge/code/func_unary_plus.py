import random
from yaql.language import specs
from yaql.language import yaqltypes
@specs.parameter('op', yaqltypes.Number())
@specs.name('#unary_operator_+')
def unary_plus(op):
    """:yaql:operator unary +

    Returns +op.

    :signature: +op
    :arg op: operand
    :argType op: number
    :returnType: number

    .. code::

        yaql> +2
        2
    """
    return +op