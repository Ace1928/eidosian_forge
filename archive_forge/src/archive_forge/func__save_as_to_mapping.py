import abc
import collections
from collections import abc as cabc
import itertools
from oslo_utils import reflection
from taskflow.types import sets
from taskflow.utils import misc
def _save_as_to_mapping(save_as):
    """Convert save_as to mapping name => index.

    Result should follow storage convention for mappings.
    """
    if save_as is None:
        return collections.OrderedDict()
    if isinstance(save_as, str):
        return collections.OrderedDict([(save_as, None)])
    elif isinstance(save_as, _sequence_types):
        return collections.OrderedDict(((key, num) for num, key in enumerate(save_as)))
    elif isinstance(save_as, _set_types):
        return collections.OrderedDict(((key, key) for key in save_as))
    else:
        raise TypeError('Atom provides parameter should be str, set or tuple/list, not %r' % save_as)