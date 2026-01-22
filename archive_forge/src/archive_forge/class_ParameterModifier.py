from collections import OrderedDict
import contextlib
import re
import types
import unittest
from absl.testing import parameterized
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
@tf_export('__internal__.test.combinations.ParameterModifier', v1=[])
class ParameterModifier:
    """Customizes the behavior of a particular parameter.

  Users should override `modified_arguments()` to modify the parameter they
  want, eg: change the value of certain parameter or filter it from the params
  passed to the test case.

  See the sample usage below, it will change any negative parameters to zero
  before it gets passed to test case.
  ```
  class NonNegativeParameterModifier(ParameterModifier):

    def modified_arguments(self, kwargs, requested_parameters):
      updates = {}
      for name, value in kwargs.items():
        if value < 0:
          updates[name] = 0
      return updates
  ```
  """
    DO_NOT_PASS_TO_THE_TEST = object()

    def __init__(self, parameter_name=None):
        """Construct a parameter modifier that may be specific to a parameter.

    Args:
      parameter_name:  A `ParameterModifier` instance may operate on a class of
        parameters or on a parameter with a particular name.  Only
        `ParameterModifier` instances that are of a unique type or were
        initialized with a unique `parameter_name` will be executed.
        See `__eq__` and `__hash__`.
    """
        self._parameter_name = parameter_name

    def modified_arguments(self, kwargs, requested_parameters):
        """Replace user-provided arguments before they are passed to a test.

    This makes it possible to adjust user-provided arguments before passing
    them to the test method.

    Args:
      kwargs:  The combined arguments for the test.
      requested_parameters: The set of parameters that are defined in the
        signature of the test method.

    Returns:
      A dictionary with updates to `kwargs`.  Keys with values set to
      `ParameterModifier.DO_NOT_PASS_TO_THE_TEST` are going to be deleted and
      not passed to the test.
    """
        del kwargs, requested_parameters
        return {}

    def __eq__(self, other):
        """Compare `ParameterModifier` by type and `parameter_name`."""
        if self is other:
            return True
        elif type(self) is type(other):
            return self._parameter_name == other._parameter_name
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """Compare `ParameterModifier` by type or `parameter_name`."""
        if self._parameter_name:
            return hash(self._parameter_name)
        else:
            return id(self.__class__)