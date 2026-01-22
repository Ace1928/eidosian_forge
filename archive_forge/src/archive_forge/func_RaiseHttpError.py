from __future__ import absolute_import
import io
import logging
import os
import random
import re
import time
import urllib
import httplib2
from oauth2client import client
from oauth2client import file as oauth2client_file
from oauth2client import tools
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.tools.value_mixin import ValueMixin
from googlecloudsdk.third_party.appengine._internal import six_subset
def RaiseHttpError(url, response_info, response_body, extra_msg=''):
    """Raise a urllib2.HTTPError based on an httplib2 response tuple."""
    if response_body is not None:
        stream = io.BytesIO()
        stream.write(response_body)
        stream.seek(0)
    else:
        stream = None
    if not extra_msg:
        msg = response_info.reason
    else:
        msg = response_info.reason + ' ' + extra_msg
    raise HTTPError(url, response_info.status, msg, response_info, stream)