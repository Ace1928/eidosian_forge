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
def WarnReservedURLs(self):
    """Generates a warning for reserved URLs.

    See the `version element documentation`_ to learn which URLs are reserved.

    .. _`version element documentation`:
       https://cloud.google.com/appengine/docs/python/config/appref#syntax
    """
    if self.url == '/form':
        logging.warning('The URL path "/form" is reserved and will not be matched.')