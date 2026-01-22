import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _add_checksum_response_metadata(self, response, response_metadata):
    checksum_context = response.get('context', {}).get('checksum', {})
    algorithm = checksum_context.get('response_algorithm')
    if algorithm:
        response_metadata['ChecksumAlgorithm'] = algorithm