from __future__ import absolute_import
import six
import json
import logging
import platform
from six.moves.urllib.parse import urlencode
from googleapiclient.errors import HttpError
def _build_query(self, params):
    """Builds a query string.

    Args:
      params: dict, the query parameters

    Returns:
      The query parameters properly encoded into an HTTP URI query string.
    """
    if self.alt_param is not None:
        params.update({'alt': self.alt_param})
    astuples = []
    for key, value in six.iteritems(params):
        if type(value) == type([]):
            for x in value:
                x = x.encode('utf-8')
                astuples.append((key, x))
        else:
            if isinstance(value, six.text_type) and callable(value.encode):
                value = value.encode('utf-8')
            astuples.append((key, value))
    return '?' + urlencode(astuples)