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
def ToDict(self):
    """Converts a `BuiltinHander` object to a dictionary.

    Returns:
      A dictionary in `{builtin_handler_name: on/off}` form
    """
    return {self.builtin_name: getattr(self, self.builtin_name)}