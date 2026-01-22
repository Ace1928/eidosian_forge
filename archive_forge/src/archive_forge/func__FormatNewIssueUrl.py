from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
import sys
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_attr_os
import six
from six.moves import range
from six.moves import urllib
def _FormatNewIssueUrl(comment):
    params = {'description': comment, 'component': six.text_type(ISSUE_TRACKER_COMPONENT)}
    return NEW_ISSUE_URL + '?' + urllib.parse.urlencode(params)