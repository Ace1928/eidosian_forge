from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import collections
import contextlib
import enum
from functools import wraps  # pylint:disable=g-importing-member
import itertools
import re
import sys
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import display
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_printer
import six
def UseRequests():
    """Returns True if using requests to make HTTP requests.

  transport/disable_requests_override is a global switch to turn off requests in
  case support is buggy. transport/opt_out_requests is an internal property
  to opt surfaces out of requests.
  """
    return UseGoogleAuth() and (not properties.VALUES.transport.opt_out_requests.GetBool()) and (not properties.VALUES.transport.disable_requests_override.GetBool())