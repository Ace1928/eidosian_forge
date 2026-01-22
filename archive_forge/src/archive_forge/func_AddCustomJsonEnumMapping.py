import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
def AddCustomJsonEnumMapping(enum_type, python_name, json_name, package=None):
    """Add a custom wire encoding for a given enum value.

    This is primarily used in generated code, to handle enum values
    which happen to be Python keywords.

    Args:
      enum_type: (messages.Enum) An enum type
      python_name: (basestring) Python name for this value.
      json_name: (basestring) JSON name to be used on the wire.
      package: (NoneType, optional) No effect, exists for legacy compatibility.
    """
    if not issubclass(enum_type, messages.Enum):
        raise exceptions.TypecheckError('Cannot set JSON enum mapping for non-enum "%s"' % enum_type)
    if python_name not in enum_type.names():
        raise exceptions.InvalidDataError('Enum value %s not a value for type %s' % (python_name, enum_type))
    field_mappings = _JSON_ENUM_MAPPINGS.setdefault(enum_type, {})
    _CheckForExistingMappings('enum', enum_type, python_name, json_name)
    field_mappings[python_name] = json_name