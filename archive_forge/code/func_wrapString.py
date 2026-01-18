def wrapString(value):
    return value if isinstance(value, DiagramItem) else Terminal(value)