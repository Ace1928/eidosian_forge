from __future__ import absolute_import
import datetime
import uuid
from googleapiclient import errors
from googleapiclient import _helpers as util
import six
def _upper_header_keys(headers):
    new_headers = {}
    for k, v in six.iteritems(headers):
        new_headers[k.upper()] = v
    return new_headers