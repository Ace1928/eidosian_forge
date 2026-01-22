import copy
import logging
import debtcollector
from oslo_config import cfg
from oslo_middleware import base
import webob.exc
@staticmethod
def _split_header_values(request, header_name):
    """Convert a comma-separated header value into a list of values."""
    values = []
    if header_name in request.headers:
        for value in request.headers[header_name].rsplit(','):
            value = value.strip()
            if value:
                values.append(value)
    return values