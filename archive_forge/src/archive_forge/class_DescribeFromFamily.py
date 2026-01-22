from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.images import flags
class DescribeFromFamily(base.DescribeCommand):
    """Describe the latest image from an image family.

  *{command}* looks up the latest image from an image family and runs a describe
  on it.
  """

    @staticmethod
    def Args(parser):
        DescribeFromFamily.DiskImageArg = flags.MakeDiskImageArg()
        DescribeFromFamily.DiskImageArg.AddArgument(parser, operation_type='describe')
        parser.add_argument('--zone', completer=completers.ZonesCompleter, help='Zone to query. Returns the latest image available in the image family for the specified zone. If not specified, returns the latest globally available image.')

    def Run(self, args):
        holder = base_classes.ComputeApiHolder(self.ReleaseTrack())
        client = holder.client
        image_ref = DescribeFromFamily.DiskImageArg.ResolveAsResource(args, holder.resources, scope_lister=compute_flags.GetDefaultScopeLister(client))
        family = image_ref.image
        if family.startswith('family/'):
            family = family[len('family/'):]
        if hasattr(args, 'zone') and args.zone:
            request = client.messages.ComputeImageFamilyViewsGetRequest(family=family, project=image_ref.project, zone=args.zone)
            return client.MakeRequests([(client.apitools_client.imageFamilyViews, 'Get', request)])[0]
        else:
            request = client.messages.ComputeImagesGetFromFamilyRequest(family=family, project=image_ref.project)
            return client.MakeRequests([(client.apitools_client.images, 'GetFromFamily', request)])[0]