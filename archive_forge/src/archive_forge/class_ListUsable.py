from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class ListUsable(base.ListCommand):
    """List subnets usable for cluster creation in a specific project.

      Usability of subnetworks for cluster creation is dependent on the IAM
      policy of the project's Google Kubernetes Engine Service Account. Use the
      `--project` flag to evaluate subnet usability in different projects. This
      list may differ from the list returned by Google Compute Engine's
      `list-usable` command, which returns subnets only usable by the caller.

      To show subnetworks shared from a Shared-VPC host project, use
      `--network-project` to specify the project that owns the subnetworks.

      ## EXAMPLES

      List all subnetworks usable for cluster creation in project `my-project`.

          $ {command} \\
            --project=my-project

      List all subnetworks existing in project `my-shared-host-project` usable
      for cluster creation in project `my-service-project`.

          $ {command} \\
             --project=my-service-project \\
             --network-project=my-shared-host-project

  """

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        parser.add_argument('--network-project', help='        The project owning the subnetworks returned. This field is translated\n        into the expression `networkProjectId=[PROJECT_ID]` and ANDed to\n        the `--filter` flag value.\n\n        Defaults to the *--project* value.\n')
        display_format = 'table({fields})'.format(fields=','.join(['subnetwork.segment(-5):label=PROJECT', 'subnetwork.segment(-3):label=REGION', 'network.segment(-1):label=NETWORK', 'subnetwork.segment(-1):label=SUBNET', 'ipCidrRange:label=RANGE', '\n        secondaryIpRanges:format="table[box](\n          rangeName:label=SECONDARY_RANGE_NAME,\n          ipCidrRange,\n          status.enum(UsableSubnetworkSecondaryRange.Status)\n        )":label=SECONDARY_RANGES\n        ']))
        parser.display_info.AddFormat(display_format)
        parser.display_info.AddUriFunc(_GetUriFunction)
        status_enum = {'UsableSubnetworkSecondaryRange.Status::enum': {'UNKNOWN': 'Unknown', 'UNUSED': 'usable for pods or services', 'IN_USE_SERVICE': 'usable for services', 'IN_USE_SHAREABLE_POD': 'usable for pods', 'IN_USE_MANAGED_POD': 'unusable'}}
        parser.display_info.AddTransforms(status_enum)

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Returns:
      Some value that we want to have printed later.
    """
        adapter = self.context['api_adapter']
        project_ref = adapter.registry.Create('container.projects', projectsId=properties.VALUES.core.project.GetOrFail())
        try:
            resp = adapter.ListUsableSubnets(project_ref, args.network_project, args.filter).subnetworks
            msg_set = set()
            for subnet in resp:
                msg = subnet.statusMessage
                if msg and msg not in msg_set:
                    msg_set.add(msg)
                    log.warning(msg)
            return resp
        except apitools_exceptions.HttpError as error:
            raise exceptions.HttpException(error, util.HTTP_ERROR_FORMAT)