from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def GetValueFunc():
    val = args[key] if key in args else None
    if val:
        return val
    raise calliope_exceptions.InvalidArgumentException('--create-disk', 'KMS cryptokey resource was not fully specified. Key [{}] must be specified.'.format(key))