from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class TapInjector(object):
    """Tap item injector."""

    def __init__(self, value, replace=False):
        self._value = value
        self._is_replacement = replace

    @property
    def value(self):
        return self._value

    @property
    def is_replacement(self):
        return self._is_replacement