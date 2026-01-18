import itertools
import json
import pkgutil
import re
from jsonschema.compat import str_types, MutableMapping, urlsplit
def types_msg(instance, types):
    """
    Create an error message for a failure to match the given types.

    If the ``instance`` is an object and contains a ``name`` property, it will
    be considered to be a description of that object and used as its type.

    Otherwise the message is simply the reprs of the given ``types``.

    """
    reprs = []
    for type in types:
        try:
            reprs.append(repr(type['name']))
        except Exception:
            reprs.append(repr(type))
    return '%r is not of type %s' % (instance, ', '.join(reprs))