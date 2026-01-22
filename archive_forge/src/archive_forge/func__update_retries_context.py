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
def _update_retries_context(self, context, attempt, success_response=None):
    retries_context = context.setdefault('retries', {})
    retries_context['attempt'] = attempt
    if 'invocation-id' not in retries_context:
        retries_context['invocation-id'] = str(uuid.uuid4())
    if success_response:
        read_timeout = context['client_config'].read_timeout
        self._set_ttl(retries_context, read_timeout, success_response)