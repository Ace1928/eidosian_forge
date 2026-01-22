import abc
import collections
import itertools
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
from oslo_utils import strutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import param_utils
from heat.engine import constraints as constr
class ParsedParameter(Parameter):
    """A template parameter with cached parsed value."""
    __slots__ = ('_parsed',)

    def __init__(self, name, schema, value=None):
        super(ParsedParameter, self).__init__(name, schema, value)
        self._parsed = None

    @property
    def parsed(self):
        if self._parsed is None:
            if self.has_value():
                if self.user_value is not None:
                    self._parsed = self.parse(self.user_value)
                else:
                    self._parsed = self.parse(self.default())
            else:
                self._parsed = self.default_parsed()
        return self._parsed