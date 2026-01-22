from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import datetime
import json
import os
import textwrap
import time
from typing import Optional
import dateutil
from googlecloudsdk.api_lib.auth import util as auth_util
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
from googlecloudsdk.core.credentials import gce as c_gce
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import times
from oauth2client import client
from oauth2client import crypt
from oauth2client import service_account
from oauth2client.contrib import gce as oauth2client_gce
from oauth2client.contrib import reauth_errors
import six
from six.moves import urllib
def _TokenExpiresWithinWindow(expiry_window, token_expiry_time, max_window_seconds=3600):
    """Determines if token_expiry_time is within expiry_window_duration.

  Calculates the amount of time between utcnow() and token_expiry_time and
  returns true, if that amount is less than the provided duration window. All
  calculations are done in number of seconds for consistency.


  Args:
    expiry_window: string, Duration representing the amount of time between
      now and token_expiry_time to compare against.
    token_expiry_time: datetime, The time when token expires.
    max_window_seconds: int, Maximum size of expiry window, in seconds.

  Raises:
    ValueError: If expiry_window is invalid or can not be parsed.

  Returns:
    True if token is expired or will expire with in the provided window,
    False otherwise.
  """
    try:
        min_expiry = times.ParseDuration(expiry_window, default_suffix='s')
        if min_expiry.total_seconds > max_window_seconds:
            raise ValueError('Invalid expiry window duration [{}]: Must be between 0s and 1h'.format(expiry_window))
    except times.Error as e:
        message = six.text_type(e).rstrip('.')
        raise ValueError('Error Parsing expiry window duration [{}]: {}'.format(expiry_window, message))
    token_expiry_time = times.LocalizeDateTime(token_expiry_time, tzinfo=dateutil.tz.tzutc())
    window_end = times.GetDateTimePlusDuration(times.Now(tzinfo=dateutil.tz.tzutc()), min_expiry)
    return token_expiry_time <= window_end