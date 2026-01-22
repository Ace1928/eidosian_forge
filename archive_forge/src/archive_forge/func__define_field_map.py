from collections.abc import Mapping, Hashable
import collections
import copy
from ..language import ast
from ..pyutils.cached_property import cached_property
from ..pyutils.ordereddict import OrderedDict
from ..utils.assert_valid_name import assert_valid_name
def _define_field_map(self):
    fields = self._fields
    if callable(fields):
        fields = fields()
    assert isinstance(fields, Mapping) and len(fields) > 0, '{} fields must be a mapping (dict / OrderedDict) with field names as keys or a function which returns such a mapping.'.format(self)
    if not isinstance(fields, (collections.OrderedDict, OrderedDict)):
        fields = OrderedDict(sorted(list(fields.items())))
    for field_name, field in fields.items():
        assert_valid_name(field_name)
    return fields