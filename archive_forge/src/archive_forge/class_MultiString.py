import collections
import operator
import re
import warnings
import abc
from debtcollector import removals
import netaddr
import rfc3986
class MultiString(String):
    """Multi-valued string."""

    def __init__(self, type_name='multi valued'):
        super(MultiString, self).__init__(type_name=type_name)
    NONE_DEFAULT = ['']

    def format_defaults(self, default, sample_default=None):
        """Return a list of formatted default values.

        """
        if sample_default is not None:
            default_list = self._formatter(sample_default)
        elif not default:
            default_list = self.NONE_DEFAULT
        else:
            default_list = self._formatter(default)
        return default_list

    def _formatter(self, value):
        return [self.quote_trailing_and_leading_space(v) for v in value]