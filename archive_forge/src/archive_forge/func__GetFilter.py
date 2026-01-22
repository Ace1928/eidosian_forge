from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.containeranalysis import util as containeranalysis_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.images import flags as image_flags
from googlecloudsdk.core import properties
def _GetFilter(self, args, holder):
    filters = ['kind = "PACKAGE_VULNERABILITY"', 'has_prefix(resource_url,"https://compute.googleapis.com/compute/")']
    if args.image:
        image_ref = self._image_arg.ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(holder.client))
        image_url = image_ref.SelfLink()
        filters.append('has_prefix(resource_url, "{}")'.format(image_url))
    return ' AND '.join(filters)