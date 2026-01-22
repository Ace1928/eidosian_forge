import datetime
import logging
import os
import threading
import time
import uuid
from botocore import parsers
from botocore.awsrequest import create_request_object
from botocore.exceptions import HTTPClientError
from botocore.history import get_global_history_recorder
from botocore.hooks import first_non_none_response
from botocore.httpchecksum import handle_checksum_body
from botocore.httpsession import URLLib3Session
from botocore.response import StreamingBody
from botocore.utils import (
def _add_modeled_error_fields(self, response_dict, parsed_response, operation_model, parser):
    error_code = parsed_response.get('Error', {}).get('Code')
    if error_code is None:
        return
    service_model = operation_model.service_model
    error_shape = service_model.shape_for_error_code(error_code)
    if error_shape is None:
        return
    modeled_parse = parser.parse(response_dict, error_shape)
    parsed_response.update(modeled_parse)