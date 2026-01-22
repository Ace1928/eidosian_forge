from __future__ import absolute_import
import six
from six.moves import http_client
from six.moves import range
from six import BytesIO, StringIO
from six.moves.urllib.parse import urlparse, urlunparse, quote, unquote
import copy
import httplib2
import json
import logging
import mimetypes
import os
import random
import socket
import time
import uuid
from email.generator import Generator
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
from email.parser import FeedParser
from googleapiclient import _helpers as util
from googleapiclient import _auth
from googleapiclient.errors import BatchError
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidChunkSizeError
from googleapiclient.errors import ResumableUploadError
from googleapiclient.errors import UnexpectedBodyError
from googleapiclient.errors import UnexpectedMethodError
from googleapiclient.model import JsonModel
def _should_retry_response(resp_status, content):
    """Determines whether a response should be retried.

  Args:
    resp_status: The response status received.
    content: The response content body.

  Returns:
    True if the response should be retried, otherwise False.
  """
    reason = None
    if resp_status >= 500:
        return True
    if resp_status == _TOO_MANY_REQUESTS:
        return True
    if resp_status == six.moves.http_client.FORBIDDEN:
        if not content:
            return False
        try:
            data = json.loads(content.decode('utf-8'))
            if isinstance(data, dict):
                error_detail_keyword = next((kw for kw in ['errors', 'status', 'message'] if kw in data['error']), '')
                if error_detail_keyword:
                    reason = data['error'][error_detail_keyword]
                    if isinstance(reason, list) and len(reason) > 0:
                        reason = reason[0]
                        if 'reason' in reason:
                            reason = reason['reason']
            else:
                reason = data[0]['error']['errors']['reason']
        except (UnicodeDecodeError, ValueError, KeyError):
            LOGGER.warning('Invalid JSON content from response: %s', content)
            return False
        LOGGER.warning('Encountered 403 Forbidden with reason "%s"', reason)
        if reason in ('userRateLimitExceeded', 'rateLimitExceeded'):
            return True
    return False