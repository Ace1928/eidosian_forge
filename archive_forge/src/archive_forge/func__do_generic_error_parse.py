import base64
import http.client
import json
import logging
import re
from botocore.compat import ETree, XMLParseError
from botocore.eventstream import EventStream, NoInitialResponseError
from botocore.utils import (
def _do_generic_error_parse(self, response):
    LOG.debug('Received a non protocol specific error response from the service, unable to populate error code and message.')
    return {'Error': {'Code': str(response['status_code']), 'Message': http.client.responses.get(response['status_code'], '')}, 'ResponseMetadata': {}}