import itertools
import json
import pkgutil
import re
from jsonschema.compat import str_types, MutableMapping, urlsplit
def uniq(container):
    """
    Check if all of a container's elements are unique.

    Successively tries first to rely that the elements are hashable, then
    falls back on them being sortable, and finally falls back on brute
    force.

    """
    try:
        return len(set((unbool(i) for i in container))) == len(container)
    except TypeError:
        try:
            sort = sorted((unbool(i) for i in container))
            sliced = itertools.islice(sort, 1, None)
            for i, j in zip(sort, sliced):
                if i == j:
                    return False
        except (NotImplementedError, TypeError):
            seen = []
            for e in container:
                e = unbool(e)
                if e in seen:
                    return False
                seen.append(e)
    return True