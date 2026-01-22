import types
import weakref
import six
from apitools.base.protorpclite import util
class FieldList(list):
    """List implementation that validates field values.

    This list implementation overrides all methods that add values in
    to a list in order to validate those new elements. Attempting to
    add or set list values that are not of the correct type will raise
    ValidationError.

    """

    def __init__(self, field_instance, sequence):
        """Constructor.

        Args:
          field_instance: Instance of field that validates the list.
          sequence: List or tuple to construct list from.
        """
        if not field_instance.repeated:
            raise FieldDefinitionError('FieldList may only accept repeated fields')
        self.__field = field_instance
        self.__field.validate(sequence)
        list.__init__(self, sequence)

    def __getstate__(self):
        """Enable pickling.

        The assigned field instance can't be pickled if it belongs to
        a Message definition (message_definition uses a weakref), so
        the Message class and field number are returned in that case.

        Returns:
          A 3-tuple containing:
            - The field instance, or None if it belongs to a Message class.
            - The Message class that the field instance belongs to, or None.
            - The field instance number of the Message class it belongs to, or
                None.

        """
        message_class = self.__field.message_definition()
        if message_class is None:
            return (self.__field, None, None)
        return (None, message_class, self.__field.number)

    def __setstate__(self, state):
        """Enable unpickling.

        Args:
          state: A 3-tuple containing:
            - The field instance, or None if it belongs to a Message class.
            - The Message class that the field instance belongs to, or None.
            - The field instance number of the Message class it belongs to, or
                None.
        """
        field_instance, message_class, number = state
        if field_instance is None:
            self.__field = message_class.field_by_number(number)
        else:
            self.__field = field_instance

    @property
    def field(self):
        """Field that validates list."""
        return self.__field

    def __setslice__(self, i, j, sequence):
        """Validate slice assignment to list."""
        self.__field.validate(sequence)
        list.__setslice__(self, i, j, sequence)

    def __setitem__(self, index, value):
        """Validate item assignment to list."""
        if isinstance(index, slice):
            self.__field.validate(value)
        else:
            self.__field.validate_element(value)
        list.__setitem__(self, index, value)

    def append(self, value):
        """Validate item appending to list."""
        self.__field.validate_element(value)
        return list.append(self, value)

    def extend(self, sequence):
        """Validate extension of list."""
        self.__field.validate(sequence)
        return list.extend(self, sequence)

    def insert(self, index, value):
        """Validate item insertion to list."""
        self.__field.validate_element(value)
        return list.insert(self, index, value)