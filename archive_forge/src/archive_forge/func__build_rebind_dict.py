import abc
import collections
from collections import abc as cabc
import itertools
from oslo_utils import reflection
from taskflow.types import sets
from taskflow.utils import misc
def _build_rebind_dict(req_args, rebind_args):
    """Build a argument remapping/rebinding dictionary.

    This dictionary allows an atom to declare that it will take a needed
    requirement bound to a given name with another name instead (mapping the
    new name onto the required name).
    """
    if rebind_args is None:
        return collections.OrderedDict()
    elif isinstance(rebind_args, (list, tuple)):
        rebind = collections.OrderedDict(zip(req_args, rebind_args))
        if len(req_args) < len(rebind_args):
            rebind.update(((a, a) for a in rebind_args[len(req_args):]))
        return rebind
    elif isinstance(rebind_args, dict):
        return rebind_args
    else:
        raise TypeError("Invalid rebind value '%s' (%s)" % (rebind_args, type(rebind_args)))