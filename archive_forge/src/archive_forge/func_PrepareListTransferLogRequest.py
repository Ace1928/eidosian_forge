from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def PrepareListTransferLogRequest(reference, max_results=None, page_token=None, message_type=None):
    """Create and populate a transfer log list request."""
    request = dict(parent=reference)
    if max_results is not None:
        if max_results > MAX_RESULTS:
            max_results = MAX_RESULTS
        request['pageSize'] = max_results
    if page_token is not None:
        request['pageToken'] = page_token
    if message_type is not None:
        if 'messageTypes:' in message_type:
            try:
                message_type = message_type.split(':')[1].split(',')
                request['messageTypes'] = message_type
            except IndexError as e:
                raise bq_error.BigqueryError('Invalid flag argument "' + message_type + '"') from e
        else:
            raise bq_error.BigqueryError('Invalid flag argument "' + message_type + '"')
    return request