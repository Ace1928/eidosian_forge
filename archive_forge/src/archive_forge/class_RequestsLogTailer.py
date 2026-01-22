from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import re
import threading
import time
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.logging import common
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import requests as creds_requests
from googlecloudsdk.core.util import encoding
import requests
class RequestsLogTailer(object):
    """LogTailer transport to make HTTP requests using requests."""

    def __init__(self):
        self.session = creds_requests.GetSession()

    def Request(self, url, cursor):
        try:
            response = self.session.request('GET', url, headers={'Range': 'bytes={0}-'.format(cursor)})
            return Response(response.status_code, response.headers, response.content)
        except requests.exceptions.RequestException as e:
            raise api_exceptions.CommunicationError('Failed to connect: %s' % e)