def validate_triad_var_name(expr: str) -> bool:
    """Check if ``expr`` is a valid Triad variable name based on Triad standard:
    it has to be a valid python identifier and it can't be purely ``_``

    .. note::

        Any valid triad var name can be used as column names without quote ` `

    :param expr: column name expression
    :return: whether it is valid
    """
    if not isinstance(expr, str) or not expr.isidentifier() or (not expr.isascii()):
        return False
    return expr.strip('_') != ''