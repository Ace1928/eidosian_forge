import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _handle_map(self, shape, value):
    parsed = {}
    key_shape = shape.key
    value_shape = shape.value
    for key, value in value.items():
        actual_key = self._parse_shape(key_shape, key)
        actual_value = self._parse_shape(value_shape, value)
        parsed[actual_key] = actual_value
    return parsed