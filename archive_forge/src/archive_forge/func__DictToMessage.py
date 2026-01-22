from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _DictToMessage(args, messages):
    """Returns the Cloud KMS crypto key name based on the values in the dict."""
    key = _DictToKmsKey(args)
    if not key:
        return None
    return messages.CustomerEncryptionKey(kmsKeyName=key.RelativeName())