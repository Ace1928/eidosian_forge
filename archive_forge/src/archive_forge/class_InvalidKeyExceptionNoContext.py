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
class InvalidKeyExceptionNoContext(InvalidKeyFileException):
    """Indicate that a particular key is bad and why."""

    def __init__(self, key, issue):
        self.key = key
        self.issue = issue
        super(InvalidKeyExceptionNoContext, self).__init__('Invalid key, [{0}] : {1}'.format(self.key, self.issue))