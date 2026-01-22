from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _GetSpecifiedKmsArgs(args):
    """Returns the first KMS related argument as a string."""
    if not args:
        return None
    specified = set()
    for keyword in _KMS_ARGS:
        if getattr(args, keyword.replace('-', '_'), None):
            specified.add('--' + keyword)
    return specified