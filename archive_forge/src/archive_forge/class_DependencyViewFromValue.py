from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
from googlecloudsdk.calliope.concepts import deps as deps_lib
from googlecloudsdk.command_lib.concepts import base
from googlecloudsdk.command_lib.concepts import exceptions
from googlecloudsdk.command_lib.concepts import names
import six
class DependencyViewFromValue(object):
    """Simple namespace for single value."""

    def __init__(self, value_getter, marshalled_dependencies=None):
        self._value_getter = value_getter
        self._marshalled_dependencies = marshalled_dependencies

    @property
    def value(self):
        """Lazy value getter.

    Returns:
      the value of the attribute, from its fallthroughs.

    Raises:
      deps_lib.AttributeNotFoundError: if the value cannot be found.
    """
        try:
            return self._value_getter()
        except TypeError:
            return self._value_getter

    @property
    def marshalled_dependencies(self):
        """Returns the marshalled dependencies or None if not marshalled."""
        return self._marshalled_dependencies