import abc
import copy
from neutron_lib import exceptions
class StringMatchingFilterObj(FilterObj):

    @property
    def is_contains(self):
        return bool(getattr(self, 'contains', False))

    @property
    def is_starts(self):
        return bool(getattr(self, 'starts', False))

    @property
    def is_ends(self):
        return bool(getattr(self, 'ends', False))