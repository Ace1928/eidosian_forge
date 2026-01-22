import base64
import calendar
import datetime
import json
import re
from xml.etree import ElementTree
from botocore import validate
from botocore.compat import formatdate
from botocore.exceptions import ParamValidationError
from botocore.utils import (
def _serialize_payload(self, partitioned, parameters, serialized, shape, shape_members):
    payload_member = shape.serialization.get('payload')
    if self._has_streaming_payload(payload_member, shape_members):
        body_payload = parameters.get(payload_member, b'')
        body_payload = self._encode_payload(body_payload)
        serialized['body'] = body_payload
    elif payload_member is not None:
        body_params = parameters.get(payload_member)
        if body_params is not None:
            serialized['body'] = self._serialize_body_params(body_params, shape_members[payload_member])
        else:
            serialized['body'] = self._serialize_empty_body()
    elif partitioned['body_kwargs']:
        serialized['body'] = self._serialize_body_params(partitioned['body_kwargs'], shape)
    elif self._requires_empty_body(shape):
        serialized['body'] = self._serialize_empty_body()