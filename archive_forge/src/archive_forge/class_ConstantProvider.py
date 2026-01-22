import copy
import logging
import os
from botocore import utils
from botocore.exceptions import InvalidConfigError
class ConstantProvider(BaseProvider):
    """This provider provides a constant value."""

    def __init__(self, value):
        self._value = value

    def __deepcopy__(self, memo):
        return ConstantProvider(copy.deepcopy(self._value, memo))

    def provide(self):
        """Provide the constant value given during initialization."""
        return self._value

    def __repr__(self):
        return 'ConstantProvider(value=%s)' % self._value