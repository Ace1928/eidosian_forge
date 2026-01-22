from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import re
from typing import Any, List, NamedTuple, Optional
from utils import bq_error
from utils import bq_id_utils
def PrepareListRequest(reference, max_results=None, page_token=None, filter_expression=None):
    """Create and populate a list request."""
    request = dict(reference)
    if max_results is not None:
        request['maxResults'] = max_results
    if filter_expression is not None:
        request['filter'] = filter_expression
    if page_token is not None:
        request['pageToken'] = page_token
    return request