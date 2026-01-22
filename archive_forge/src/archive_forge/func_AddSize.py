from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
def AddSize(instance_ref, args, patch_request):
    """Python hook to add size update to the redis instance update request."""
    if args.IsSpecified('size'):
        _WarnForDestructiveSizeUpdate(instance_ref, patch_request.instance)
        patch_request.instance.memorySizeGb = args.size
        patch_request = AddFieldToUpdateMask('memory_size_gb', patch_request)
    return patch_request