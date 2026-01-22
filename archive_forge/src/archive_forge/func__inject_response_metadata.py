import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _inject_response_metadata(self, parsed, headers):
    if 'x-amzn-requestid' in headers:
        parsed.setdefault('ResponseMetadata', {})['RequestId'] = headers['x-amzn-requestid']