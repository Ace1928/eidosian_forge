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
@base.ReleaseTracks(base.ReleaseTrack.GA)
class ListTagsGA(base.ListCommand):
    """List tags and digests for the specified image."""
    detailed_help = {'DESCRIPTION': '          The container images list-tags command of gcloud lists metadata about\n          tags and digests for the specified container image. Images must be\n          hosted by the Google Container Registry.\n      ', 'EXAMPLES': '          List the tags in a specified image:\n\n            $ {command} gcr.io/myproject/myimage\n\n          To receive the full, JSON-formatted output (with untruncated digests):\n\n            $ {command} gcr.io/myproject/myimage --format=json\n\n          To list digests without corresponding tags:\n\n            $ {command} gcr.io/myproject/myimage --filter="NOT tags:*"\n\n          To list images that have a tag with the value \'30e5504145\':\n\n            $ gcloud container images list-tags --filter="\'tags:30e5504145\'"\n\n          The last example encloses the filter expression in single quotes\n          because the value \'30e5504145\' could be interpreted as a number in\n          scientific notation.\n\n      '}

    @staticmethod
    def Args(parser):
        """Register flags for this command.

    Args:
      parser: An argparse.ArgumentParser-like object. It is mocked out in order
        to capture some information, but behaves like an ArgumentParser.
    """
        flags.AddImagePositional(parser, verb='list tags for')
        base.SORT_BY_FLAG.SetDefault(parser, _DEFAULT_SORT_BY)
        base.URI_FLAG.RemoveFromParser(parser)
        parser.display_info.AddFormat(_TAGS_FORMAT)

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
        repository = util.ValidateRepositoryPath(args.image_name)
        http_obj = util.Http()
        with util.WrapExpectedDockerlessErrors(repository):
            with docker_image.FromRegistry(basic_creds=util.CredentialProvider(), name=repository, transport=http_obj) as image:
                manifests = image.manifests()
                return util.TransformManifests(manifests, repository)