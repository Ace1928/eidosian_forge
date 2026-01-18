from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os
import re
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
@classmethod
def validate_url_string(cls, url_string, scheme):
    AzureUrl.is_valid_scheme(scheme)
    if not (AZURE_DOMAIN in url_string and AzureUrl.is_valid_scheme(scheme)):
        raise errors.InvalidUrlError('Invalid Azure URL: "{}"'.format(url_string))