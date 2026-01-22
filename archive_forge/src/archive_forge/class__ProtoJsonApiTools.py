import base64
import collections
import datetime
import json
import six
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import protojson
from apitools.base.py import exceptions
class _ProtoJsonApiTools(protojson.ProtoJson):
    """JSON encoder used by apitools clients."""
    _INSTANCE = None

    @classmethod
    def Get(cls):
        if cls._INSTANCE is None:
            cls._INSTANCE = cls()
        return cls._INSTANCE

    def decode_message(self, message_type, encoded_message):
        if message_type in _CUSTOM_MESSAGE_CODECS:
            return _CUSTOM_MESSAGE_CODECS[message_type].decoder(encoded_message)
        result = _DecodeCustomFieldNames(message_type, encoded_message)
        result = super(_ProtoJsonApiTools, self).decode_message(message_type, result)
        result = _ProcessUnknownEnums(result, encoded_message)
        result = _ProcessUnknownMessages(result, encoded_message)
        return _DecodeUnknownFields(result, encoded_message)

    def decode_field(self, field, value):
        """Decode the given JSON value.

        Args:
          field: a messages.Field for the field we're decoding.
          value: a python value we'd like to decode.

        Returns:
          A value suitable for assignment to field.
        """
        for decoder in _GetFieldCodecs(field, 'decoder'):
            result = decoder(field, value)
            value = result.value
            if result.complete:
                return value
        if isinstance(field, messages.MessageField):
            field_value = self.decode_message(field.message_type, json.dumps(value))
        elif isinstance(field, messages.EnumField):
            value = GetCustomJsonEnumMapping(field.type, json_name=value) or value
            try:
                field_value = super(_ProtoJsonApiTools, self).decode_field(field, value)
            except messages.DecodeError:
                if not isinstance(value, six.string_types):
                    raise
                field_value = None
        else:
            field_value = super(_ProtoJsonApiTools, self).decode_field(field, value)
        return field_value

    def encode_message(self, message):
        if isinstance(message, messages.FieldList):
            return '[%s]' % ', '.join((self.encode_message(x) for x in message))
        if type(message) in _CUSTOM_MESSAGE_CODECS:
            return _CUSTOM_MESSAGE_CODECS[type(message)].encoder(message)
        message = _EncodeUnknownFields(message)
        result = super(_ProtoJsonApiTools, self).encode_message(message)
        result = _EncodeCustomFieldNames(message, result)
        return json.dumps(json.loads(result), sort_keys=True)

    def encode_field(self, field, value):
        """Encode the given value as JSON.

        Args:
          field: a messages.Field for the field we're encoding.
          value: a value for field.

        Returns:
          A python value suitable for json.dumps.
        """
        for encoder in _GetFieldCodecs(field, 'encoder'):
            result = encoder(field, value)
            value = result.value
            if result.complete:
                return value
        if isinstance(field, messages.EnumField):
            if field.repeated:
                remapped_value = [GetCustomJsonEnumMapping(field.type, python_name=e.name) or e.name for e in value]
            else:
                remapped_value = GetCustomJsonEnumMapping(field.type, python_name=value.name)
            if remapped_value:
                return remapped_value
        if isinstance(field, messages.MessageField) and (not isinstance(field, message_types.DateTimeField)):
            value = json.loads(self.encode_message(value))
        return super(_ProtoJsonApiTools, self).encode_field(field, value)