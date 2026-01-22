from __future__ import absolute_import
import six
import json
import logging
import platform
from six.moves.urllib.parse import urlencode
from googleapiclient.errors import HttpError
def _log_response(self, resp, content):
    """Logs debugging information about the response if requested."""
    if dump_request_response:
        LOGGER.info('--response-start--')
        for h, v in six.iteritems(resp):
            LOGGER.info('%s: %s', h, v)
        if content:
            LOGGER.info(content)
        LOGGER.info('--response-end--')