import itertools
import json
import pkgutil
import re
from jsonschema.compat import str_types, MutableMapping, urlsplit
def unbool(element, true=object(), false=object()):
    """
    A hack to make True and 1 and False and 0 unique for ``uniq``.

    """
    if element is True:
        return true
    elif element is False:
        return false
    return element