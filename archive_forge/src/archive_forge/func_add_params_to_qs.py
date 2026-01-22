from __future__ import absolute_import, unicode_literals
import collections
import datetime
import logging
import re
import sys
import time
def add_params_to_qs(query, params):
    """Extend a query with a list of two-tuples."""
    if isinstance(params, dict):
        params = params.items()
    queryparams = urlparse.parse_qsl(query, keep_blank_values=True)
    queryparams.extend(params)
    return urlencode(queryparams)