from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import heapq
import sys
from containerregistry.client.v2_2 import docker_image
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import exceptions
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.BETA)
class ListTagsALPHAandBETA(ListTagsGA, base.ListCommand):
    """List tags and digests for the specified image."""

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        super(ListTagsALPHAandBETA, ListTagsALPHAandBETA).Args(parser)
        parser.add_argument('--show-occurrences', action='store_true', default=True, help='Whether to show summaries of the various Occurrence types.')
        parser.add_argument('--occurrence-filter', default=' OR '.join(['kind = "{kind}"'.format(kind=x) for x in _DEFAULT_KINDS]), help='A filter for the Occurrences which will be summarized.')
        parser.add_argument('--show-occurrences-from', type=arg_parsers.BoundedInt(1, sys.maxsize, unlimited=True), default=_DEFAULT_SHOW_OCCURRENCES_FROM, help='How many of the most recent images for which to summarize Occurences.')

    def Run(self, args):
        """This is what gets called when the user runs this command.

    Args:
      args: an argparse namespace. All the arguments that were provided to this
        command invocation.

    Raises:
      ArgumentError: If the user provided the flag --show-occurrences-from but
        --show-occurrences=False.
      InvalidImageNameError: If the user specified an invalid image name.
    Returns:
      Some value that we want to have printed later.
    """
        if args.IsSpecified('show_occurrences_from') and (not args.show_occurrences):
            raise ArgumentError('--show-occurrences-from may only be set if --show-occurrences=True')
        repository = util.ValidateRepositoryPath(args.image_name)
        http_obj = util.Http()
        with util.WrapExpectedDockerlessErrors(repository):
            with docker_image.FromRegistry(basic_creds=util.CredentialProvider(), name=repository, transport=http_obj) as image:
                manifests = image.manifests()
                most_recent_resource_urls = None
                occ_filter = filter_util.ContainerAnalysisFilter()
                occ_filter.WithCustomFilter(args.occurrence_filter)
                occ_filter.WithResourcePrefixes(['https://{}'.format(repository)])
                if args.show_occurrences_from:
                    most_recent_resource_urls = ['https://%s@%s' % (args.image_name, k) for k in heapq.nlargest(args.show_occurrences_from, manifests, key=lambda k: manifests[k]['timeCreatedMs'])]
                    occ_filter.WithResources(most_recent_resource_urls)
                return util.TransformManifests(manifests, repository, show_occurrences=args.show_occurrences, occurrence_filter=occ_filter)