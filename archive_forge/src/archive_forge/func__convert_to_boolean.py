from collections.abc import MutableMapping
from collections import ChainMap as _ChainMap
import functools
import io
import itertools
import os
import re
import sys
import warnings
def _convert_to_boolean(self, value):
    """Return a boolean value translating from other types if necessary.
        """
    if value.lower() not in self.BOOLEAN_STATES:
        raise ValueError('Not a boolean: %s' % value)
    return self.BOOLEAN_STATES[value.lower()]