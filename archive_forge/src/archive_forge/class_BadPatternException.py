from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import json
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
class BadPatternException(InvalidKeyFileException):
    """A (e.g.) url pattern is bad and why."""

    def __init__(self, pattern_type, pattern):
        self.pattern_type = pattern_type
        self.pattern = pattern
        super(BadPatternException, self).__init__('Invalid value for [{0}] pattern: [{1}]'.format(self.pattern_type, self.pattern))