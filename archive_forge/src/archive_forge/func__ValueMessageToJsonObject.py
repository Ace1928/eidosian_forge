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
def _ValueMessageToJsonObject(self, message):
    """Converts Value message according to Proto3 JSON Specification."""
    which = message.WhichOneof('kind')
    if which is None or which == 'null_value':
        return None
    if which == 'list_value':
        return self._ListValueMessageToJsonObject(message.list_value)
    if which == 'number_value':
        value = message.number_value
        if math.isinf(value):
            raise ValueError('Fail to serialize Infinity for Value.number_value, which would parse as string_value')
        if math.isnan(value):
            raise ValueError('Fail to serialize NaN for Value.number_value, which would parse as string_value')
    else:
        value = getattr(message, which)
    oneof_descriptor = message.DESCRIPTOR.fields_by_name[which]
    return self._FieldToJsonObject(oneof_descriptor, value)