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
class BadKeyTypeException(InvalidKeyFileException):
    """A key type is bad and why."""

    def __init__(self, key_type, explanation=''):
        self.key_type = key_type
        msg = 'Invalid key type [{0}]'.format(self.key_type)
        if explanation:
            msg += ': ' + explanation
        msg += '.'
        super(BadKeyTypeException, self).__init__(msg)