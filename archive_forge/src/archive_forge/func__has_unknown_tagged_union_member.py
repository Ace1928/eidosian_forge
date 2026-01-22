import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _has_unknown_tagged_union_member(self, shape, value):
    if shape.is_tagged_union:
        cleaned_value = value.copy()
        cleaned_value.pop('__type', None)
        if len(cleaned_value) != 1:
            error_msg = 'Invalid service response: %s must have one and only one member set.'
            raise ResponseParserError(error_msg % shape.name)
        tag = self._get_first_key(cleaned_value)
        if tag not in shape.members:
            msg = 'Received a tagged union response with member unknown to client: %s. Please upgrade SDK for full response support.'
            LOG.info(msg % tag)
            return True
    return False