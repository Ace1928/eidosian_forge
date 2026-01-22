import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _is_generic_error_response(self, response):
    if response['status_code'] >= 500:
        if 'body' not in response or response['body'] is None:
            return True
        body = response['body'].strip()
        return body.startswith(b'<html>') or not body