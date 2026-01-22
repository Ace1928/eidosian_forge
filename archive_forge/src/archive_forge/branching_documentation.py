from yaql.language import specs
from yaql.language import yaqltypes
:yaql:coalesce

    Returns the first predicate which evaluates to non-null value. Returns null
    if no arguments are provided or if all of them are null.

    :signature: coalesce([args])
    :arg [args]: input arguments
    :argType [args]: chain of any types
    :returnType: any

    .. code::

        yaql> coalesce(null)
        null
        yaql> coalesce(null, [1, 2, 3][0], "abc")
        1
        yaql> coalesce(null, false, 1)
        false
    