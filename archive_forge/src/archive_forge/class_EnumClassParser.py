from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import csv
import io
import string
from absl.flags import _helpers
import six
class EnumClassParser(ArgumentParser):
    """Parser of an Enum class member."""

    def __init__(self, enum_class, case_sensitive=True):
        """Initializes EnumParser.

    Args:
      enum_class: class, the Enum class with all possible flag values.
      case_sensitive: bool, whether or not the enum is to be case-sensitive. If
        False, all member names must be unique when case is ignored.

    Raises:
      TypeError: When enum_class is not a subclass of Enum.
      ValueError: When enum_class is empty.
    """
        import enum
        if not issubclass(enum_class, enum.Enum):
            raise TypeError('{} is not a subclass of Enum.'.format(enum_class))
        if not enum_class.__members__:
            raise ValueError('enum_class cannot be empty, but "{}" is empty.'.format(enum_class))
        if not case_sensitive:
            members = collections.Counter((name.lower() for name in enum_class.__members__))
            duplicate_keys = {member for member, count in members.items() if count > 1}
            if duplicate_keys:
                raise ValueError('Duplicate enum values for {} using case_sensitive=False'.format(duplicate_keys))
        super(EnumClassParser, self).__init__()
        self.enum_class = enum_class
        self._case_sensitive = case_sensitive
        if case_sensitive:
            self._member_names = tuple(enum_class.__members__)
        else:
            self._member_names = tuple((name.lower() for name in enum_class.__members__))

    @property
    def member_names(self):
        """The accepted enum names, in lowercase if not case sensitive."""
        return self._member_names

    def parse(self, argument):
        """Determines validity of argument and returns the correct element of enum.

    Args:
      argument: str or Enum class member, the supplied flag value.

    Returns:
      The first matching Enum class member in Enum class.

    Raises:
      ValueError: Raised when argument didn't match anything in enum.
    """
        if isinstance(argument, self.enum_class):
            return argument
        elif not isinstance(argument, six.string_types):
            raise ValueError('{} is not an enum member or a name of a member in {}'.format(argument, self.enum_class))
        key = EnumParser(self._member_names, case_sensitive=self._case_sensitive).parse(argument)
        if self._case_sensitive:
            return self.enum_class[key]
        else:
            return next((value for name, value in self.enum_class.__members__.items() if name.lower() == key.lower()))

    def flag_type(self):
        """See base class."""
        return 'enum class'