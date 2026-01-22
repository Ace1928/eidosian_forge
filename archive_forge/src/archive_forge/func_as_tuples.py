from .. import debug
def as_tuples(obj):
    """Ensure that the object and any referenced objects are plain tuples.

    :param obj: a list, tuple or StaticTuple
    :return: a plain tuple instance, with all children also being tuples.
    """
    result = []
    for item in obj:
        if isinstance(item, (tuple, list, StaticTuple)):
            item = as_tuples(item)
        result.append(item)
    return tuple(result)