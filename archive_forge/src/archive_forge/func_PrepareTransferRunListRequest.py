from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def PrepareTransferRunListRequest(reference, run_attempt, max_results=None, page_token=None, states=None):
    """Create and populate a transfer run list request."""
    request = dict(parent=reference)
    request['runAttempt'] = run_attempt
    if max_results is not None:
        if max_results > MAX_RESULTS:
            max_results = MAX_RESULTS
        request['pageSize'] = max_results
    if states is not None:
        if 'states:' in states:
            try:
                states = states.split(':')[1].split(',')
                request['states'] = states
            except IndexError as e:
                raise bq_error.BigqueryError('Invalid flag argument "' + states + '"') from e
        else:
            raise bq_error.BigqueryError('Invalid flag argument "' + states + '"')
    if page_token is not None:
        request['pageToken'] = page_token
    return request