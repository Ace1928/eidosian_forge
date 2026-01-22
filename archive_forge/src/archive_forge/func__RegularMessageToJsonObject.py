import base64
from collections import OrderedDict
import json
import math
from operator import methodcaller
import re
from google.protobuf import descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database
from google.protobuf.internal import type_checkers
def _RegularMessageToJsonObject(self, message, js):
    """Converts normal message according to Proto3 JSON Specification."""
    fields = message.ListFields()
    try:
        for field, value in fields:
            if self.preserving_proto_field_name:
                name = field.name
            else:
                name = field.json_name
            if _IsMapEntry(field):
                v_field = field.message_type.fields_by_name['value']
                js_map = {}
                for key in value:
                    if isinstance(key, bool):
                        if key:
                            recorded_key = 'true'
                        else:
                            recorded_key = 'false'
                    else:
                        recorded_key = str(key)
                    js_map[recorded_key] = self._FieldToJsonObject(v_field, value[key])
                js[name] = js_map
            elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                js[name] = [self._FieldToJsonObject(field, k) for k in value]
            elif field.is_extension:
                name = '[%s]' % field.full_name
                js[name] = self._FieldToJsonObject(field, value)
            else:
                js[name] = self._FieldToJsonObject(field, value)
        if self.always_print_fields_with_no_presence:
            message_descriptor = message.DESCRIPTOR
            for field in message_descriptor.fields:
                if self.always_print_fields_with_no_presence and field.has_presence:
                    continue
                if self.preserving_proto_field_name:
                    name = field.name
                else:
                    name = field.json_name
                if name in js:
                    continue
                if _IsMapEntry(field):
                    js[name] = {}
                elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
                    js[name] = []
                else:
                    js[name] = self._FieldToJsonObject(field, field.default_value)
    except ValueError as e:
        raise SerializeToJsonError('Failed to serialize {0} field: {1}.'.format(field.name, e)) from e
    return js