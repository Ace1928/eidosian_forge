import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _handle_structure(self, shape, value):
    final_parsed = {}
    if shape.is_document_type:
        final_parsed = value
    else:
        member_shapes = shape.members
        if value is None:
            return None
        final_parsed = {}
        if self._has_unknown_tagged_union_member(shape, value):
            tag = self._get_first_key(value)
            return self._handle_unknown_tagged_union_member(tag)
        for member_name in member_shapes:
            member_shape = member_shapes[member_name]
            json_name = member_shape.serialization.get('name', member_name)
            raw_value = value.get(json_name)
            if raw_value is not None:
                final_parsed[member_name] = self._parse_shape(member_shapes[member_name], raw_value)
    return final_parsed