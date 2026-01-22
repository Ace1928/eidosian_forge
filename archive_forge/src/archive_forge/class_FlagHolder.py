from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import itertools
import logging
import os
import sys
from xml.dom import minidom
from absl.flags import _exceptions
from absl.flags import _flag
from absl.flags import _helpers
from absl.flags import _validators_classes
import six
class FlagHolder(_Base):
    """Holds a defined flag.

  This facilitates a cleaner api around global state. Instead of

  ```
  flags.DEFINE_integer('foo', ...)
  flags.DEFINE_integer('bar', ...)
  ...
  def method():
    # prints parsed value of 'bar' flag
    print(flags.FLAGS.foo)
    # runtime error due to typo or possibly bad coding style.
    print(flags.FLAGS.baz)
  ```

  it encourages code like

  ```
  FOO_FLAG = flags.DEFINE_integer('foo', ...)
  BAR_FLAG = flags.DEFINE_integer('bar', ...)
  ...
  def method():
    print(FOO_FLAG.value)
    print(BAR_FLAG.value)
  ```

  since the name of the flag appears only once in the source code.
  """

    def __init__(self, flag_values, flag, ensure_non_none_value=False):
        """Constructs a FlagHolder instance providing typesafe access to flag.

    Args:
      flag_values: The container the flag is registered to.
      flag: The flag object for this flag.
      ensure_non_none_value: Is the value of the flag allowed to be None.
    """
        self._flagvalues = flag_values
        self._name = flag.name
        self._ensure_non_none_value = ensure_non_none_value

    def __eq__(self, other):
        raise TypeError("unsupported operand type(s) for ==: '{0}' and '{1}' (did you mean to use '{0}.value' instead?)".format(type(self).__name__, type(other).__name__))

    def __bool__(self):
        raise TypeError("bool() not supported for instances of type '{0}' (did you mean to use '{0}.value' instead?)".format(type(self).__name__))
    __nonzero__ = __bool__

    @property
    def name(self):
        return self._name

    @property
    def value(self):
        """Returns the value of the flag.

    If _ensure_non_none_value is True, then return value is not None.

    Raises:
      UnparsedFlagAccessError: if flag parsing has not finished.
      IllegalFlagValueError: if value is None unexpectedly.
    """
        val = getattr(self._flagvalues, self._name)
        if self._ensure_non_none_value and val is None:
            raise _exceptions.IllegalFlagValueError('Unexpected None value for flag %s' % self._name)
        return val

    @property
    def default(self):
        """Returns the default value of the flag."""
        return self._flagvalues[self._name].default

    @property
    def present(self):
        """Returns True if the flag was parsed from command-line flags."""
        return bool(self._flagvalues[self._name].present)