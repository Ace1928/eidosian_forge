from __future__ import absolute_import
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.third_party.appengine._internal import six_subset
class MalformedDosConfiguration(Exception):
    """Configuration file for DOS API is malformed."""