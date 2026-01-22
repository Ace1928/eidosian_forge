import itertools
import collections
def __makeDict(headers, array, default=None):
    result = {}
    returndefaultvalue = None
    if len(headers) == 1:
        result.update(dict(zip(headers[0], array)))
        defaultvalue = default
    else:
        for i, h in enumerate(headers[0]):
            result[h], defaultvalue = __makeDict(headers[1:], array[i], default)
    if default is not None:
        f = lambda: defaultvalue
        defresult = collections.defaultdict(f)
        defresult.update(result)
        result = defresult
        returndefaultvalue = collections.defaultdict(f)
    return (result, returndefaultvalue)