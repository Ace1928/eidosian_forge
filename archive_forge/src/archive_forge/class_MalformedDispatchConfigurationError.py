from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
class MalformedDispatchConfigurationError(Error):
    """Configuration file for dispatch is malformed."""