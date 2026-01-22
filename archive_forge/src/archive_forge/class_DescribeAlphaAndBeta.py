from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from containerregistry.client import docker_name
from googlecloudsdk.api_lib.container.images import container_data_util
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class DescribeAlphaAndBeta(Describe):
    """Lists container analysis data for a given image.

  Lists container analysis data for a valid image.

  ## EXAMPLES

  Describe the specified image:

    $ {command} gcr.io/myproject/myimage@digest

          Or:

    $ {command} gcr.io/myproject/myimage:tag

  Find the digest for a tag:

    $ {command} gcr.io/myproject/myimage:tag \\
      --format="value(image_summary.digest)"

          Or:

    $ {command} gcr.io/myproject/myimage:tag \\
      --format="value(image_summary.fully_qualified_digest)"

  See package vulnerabilities found by the Container Analysis API for the
  specified image:

    $ {command} gcr.io/myproject/myimage@digest --show-package-vulnerability
  """

    @staticmethod
    def Args(parser):
        _CommonArgs(parser)
        parser.add_argument('--metadata-filter', default='', help='Additional filter to fetch metadata for a given fully qualified image reference.')
        parser.add_argument('--show-build-details', action='store_true', help='Include build metadata in the output.')
        parser.add_argument('--show-package-vulnerability', action='store_true', help='Include vulnerability metadata in the output.')
        parser.add_argument('--show-image-basis', action='store_true', help='Include base image metadata in the output.')
        parser.add_argument('--show-deployment', action='store_true', help='Include deployment metadata in the output.')
        parser.add_argument('--show-all-metadata', action='store_true', help='Include all metadata in the output.')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Raises:
      InvalidImageNameError: If the user specified an invalid image name.
    Returns:
      Some value that we want to have printed later.
    """
        filter_kinds = []
        if args.show_build_details:
            filter_kinds.append('BUILD')
        if args.show_package_vulnerability:
            filter_kinds.append('VULNERABILITY')
            filter_kinds.append('DISCOVERY')
        if args.show_image_basis:
            filter_kinds.append('IMAGE')
        if args.show_deployment:
            filter_kinds.append('DEPLOYMENT')
        if args.show_all_metadata:
            filter_kinds = _DEFAULT_KINDS
        if filter_kinds or args.metadata_filter:
            f = filter_util.ContainerAnalysisFilter()
            f.WithKinds(filter_kinds)
            f.WithCustomFilter(args.metadata_filter)
            with util.WrapExpectedDockerlessErrors(args.image_name):
                img_name = MaybeConvertToGCR(util.GetDigestFromName(args.image_name))
                f.WithResources(['https://{}'.format(img_name)])
                data = util.TransformContainerAnalysisData(img_name, f)
                if not data.build_details_summary.build_details and (not args.show_build_details) and (not args.show_all_metadata):
                    del data.build_details_summary
                if not data.package_vulnerability_summary.vulnerabilities and (not args.show_package_vulnerability) and (not args.show_all_metadata):
                    del data.package_vulnerability_summary
                if not data.discovery_summary.discovery and (not args.show_package_vulnerability) and (not args.show_all_metadata):
                    del data.discovery_summary
                if not data.image_basis_summary.base_images and (not args.show_image_basis) and (not args.show_all_metadata):
                    del data.image_basis_summary
                if not data.deployment_summary.deployments and (not args.show_deployment) and (not args.show_all_metadata):
                    del data.deployment_summary
                return data
        else:
            with util.WrapExpectedDockerlessErrors(args.image_name):
                img_name = MaybeConvertToGCR(util.GetDigestFromName(args.image_name))
                return container_data_util.ContainerData(registry=img_name.registry, repository=img_name.repository, digest=img_name.digest)