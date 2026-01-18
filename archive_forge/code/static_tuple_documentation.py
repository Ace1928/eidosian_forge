from .. import debug
Ensure that the object and any referenced objects are plain tuples.

    :param obj: a list, tuple or StaticTuple
    :return: a plain tuple instance, with all children also being tuples.
    