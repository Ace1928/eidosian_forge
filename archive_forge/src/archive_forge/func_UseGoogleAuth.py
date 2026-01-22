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
def UseGoogleAuth():
    """Returns True if using google-auth to authenticate the http request.

  auth/disable_load_google_auth is a global switch to turn off google-auth in
  case google-auth is crashing. auth/opt_out_google_auth is an internal property
  to opt-out a surface.
  """
    return not (properties.VALUES.auth.opt_out_google_auth.GetBool() or properties.VALUES.auth.disable_load_google_auth.GetBool())