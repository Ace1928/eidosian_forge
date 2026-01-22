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
def GetHandler(self):
    """Gets the handler for a mapping.

    Returns:
      The value of the handler, as determined by the handler ID attribute.
    """
    return getattr(self, self.GetHandlerType())