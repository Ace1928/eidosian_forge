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
class VmHealthCheck(HealthCheck):
    """Class representing the configuration of the VM health check.

  Note:
      This class is deprecated and will be removed in a future release. Use
      `HealthCheck` instead.
  """
    pass