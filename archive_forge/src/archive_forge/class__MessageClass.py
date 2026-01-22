import types
import weakref
import six
from apitools.base.protorpclite import util
class _MessageClass(_DefinitionClass):
    """Meta-class used for defining the Message base class.

    For more details about Message classes, see the Message class docstring.
    Information contained there may help understanding this class.

    Meta-class enables very specific behavior for any defined Message
    class. All attributes defined on an Message sub-class must be
    field instances, Enum class definitions or other Message class
    definitions. Each field attribute defined on an Message sub-class
    is added to the set of field definitions and the attribute is
    translated in to a slot. It also ensures that only one level of
    Message class hierarchy is possible. In other words it is not
    possible to declare sub-classes of sub-classes of Message.

    This class also defines some functions in order to restrict the
    behavior of the Message class and its sub-classes. It is not
    possible to change the behavior of the Message class in later
    classes since any new classes may be defined with only field,
    Enums and Messages, and no methods.

    """

    def __new__(cls, name, bases, dct):
        """Create new Message class instance.

        The __new__ method of the _MessageClass type is overridden so as to
        allow the translation of Field instances to slots.
        """
        by_number = {}
        by_name = {}
        variant_map = {}
        if bases != (object,):
            if bases != (Message,):
                raise MessageDefinitionError('Message types may only inherit from Message')
            enums = []
            messages = []
            for key, field in dct.items():
                if key in _RESERVED_ATTRIBUTE_NAMES:
                    continue
                if isinstance(field, type) and issubclass(field, Enum):
                    enums.append(key)
                    continue
                if isinstance(field, type) and issubclass(field, Message) and (field is not Message):
                    messages.append(key)
                    continue
                if type(field) is Field or not isinstance(field, Field):
                    raise MessageDefinitionError('May only use fields in message definitions.  Found: %s = %s' % (key, field))
                if field.number in by_number:
                    raise DuplicateNumberError('Field with number %d declared more than once in %s' % (field.number, name))
                field.name = key
                by_name[key] = field
                by_number[field.number] = field
            if enums:
                dct['__enums__'] = sorted(enums)
            if messages:
                dct['__messages__'] = sorted(messages)
        dct['_Message__by_number'] = by_number
        dct['_Message__by_name'] = by_name
        return _DefinitionClass.__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct):
        """Initializer required to assign references to new class."""
        if bases != (object,):
            for v in dct.values():
                if isinstance(v, _DefinitionClass) and v is not Message:
                    v._message_definition = weakref.ref(cls)
            for field in cls.all_fields():
                field._message_definition = weakref.ref(cls)
        _DefinitionClass.__init__(cls, name, bases, dct)