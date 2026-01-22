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
def MaybeConvertToGCR(image_name):
    """Converts gcr.io repos on AR from pkg.dev->gcr.io.

  Args:
    image_name: Image to convert to GCR.

  Returns:
    The same image_name, but maybe in GCR format.
  """
    if 'pkg.dev' not in image_name.registry:
        return image_name
    matches = re.match(GCR_REPO_REGEX, image_name.repository)
    if not matches:
        return image_name
    messages = ar_requests.GetMessages()
    settings = ar_requests.GetProjectSettings(matches.group('project'))
    if settings.legacyRedirectionState == messages.ProjectSettings.LegacyRedirectionStateValueValuesEnum.REDIRECTION_FROM_GCR_IO_DISABLED:
        log.warning('gcr.io repositories in Artifact Registry are only scanned if redirected. Redirect this project before checking scanning results')
        return image_name
    log.warning('Container Analysis API uses the gcr.io hostname for scanning results of gcr.io repositories. Using https://{}/{} instead...'.format(matches.group('repo'), matches.group('project')))
    return docker_name.Digest('{registry}/{repository}@{sha256}'.format(registry=matches.group('repo'), repository='{}/{}'.format(matches.group('project'), matches.group('image')), sha256=image_name.digest))