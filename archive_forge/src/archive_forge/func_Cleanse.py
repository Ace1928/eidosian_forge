import collections
import math
def Cleanse(obj, encoding='utf-8'):
    """Makes Python object appropriate for JSON serialization.

    - Replaces instances of Infinity/-Infinity/NaN with strings.
    - Turns byte strings into unicode strings.
    - Turns sets into sorted lists.
    - Turns tuples into lists.

    Args:
      obj: Python data structure.
      encoding: Charset used to decode byte strings.

    Returns:
      Unicode JSON data structure.
    """
    if isinstance(obj, int):
        return obj
    elif isinstance(obj, float):
        if obj == _INFINITY:
            return 'Infinity'
        elif obj == _NEGATIVE_INFINITY:
            return '-Infinity'
        elif math.isnan(obj):
            return 'NaN'
        else:
            return obj
    elif isinstance(obj, bytes):
        return obj.decode(encoding)
    elif isinstance(obj, (list, tuple)):
        return [Cleanse(i, encoding) for i in obj]
    elif isinstance(obj, set):
        return [Cleanse(i, encoding) for i in sorted(obj)]
    elif isinstance(obj, dict):
        return collections.OrderedDict(((Cleanse(k, encoding), Cleanse(v, encoding)) for k, v in obj.items()))
    else:
        return obj