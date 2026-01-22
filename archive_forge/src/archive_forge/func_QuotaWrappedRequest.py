from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope import base
from googlecloudsdk.core import context_aware
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.credentials import creds as core_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import store
from googlecloudsdk.core.util import files
from oauth2client import client
import six
from google.auth import exceptions as google_auth_exceptions
def QuotaWrappedRequest(self, http_client, quota_project):
    """Returns a request method which adds the quota project header."""
    handlers = [transport.Handler(transport.SetHeader('X-Goog-User-Project', quota_project))]
    self.WrapRequest(http_client, handlers)
    return http_client.request