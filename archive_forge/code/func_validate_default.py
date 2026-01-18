import types
import weakref
import six
from apitools.base.protorpclite import util
def validate_default(self, value):
    """Validate default value assigned to field.

        Args:
          value: Value to validate.

        Returns:
          the value in casted in the correct type.

        Raises:
          ValidationError if value is not expected type.
        """
    return self.__validate(value, self.validate_default_element)