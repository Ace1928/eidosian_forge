from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
import os
import re
import string
import sys
import wsgiref.util
from googlecloudsdk.third_party.appengine.api import appinfo_errors
from googlecloudsdk.third_party.appengine.api import backendinfo
from googlecloudsdk.third_party.appengine._internal import six_subset
def EnsureAsciiBytes(s, err):
    """Ensure s contains only ASCII-safe characters; return it as bytes-type.

  Arguments:
    s: the string or bytes to check
    err: the error to raise if not good.
  Raises:
    err if it's not ASCII-safe.
  Returns:
    s as a byte string
  """
    try:
        return s.encode('ascii')
    except UnicodeEncodeError:
        raise err
    except UnicodeDecodeError:
        raise err
    except AttributeError:
        try:
            return s.decode('ascii').encode('ascii')
        except UnicodeDecodeError:
            raise err