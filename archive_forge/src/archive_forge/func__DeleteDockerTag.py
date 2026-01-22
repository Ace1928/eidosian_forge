from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_session
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
import six
def _DeleteDockerTag(self, tag, digests, http_obj):
    docker_session.Delete(creds=util.CredentialProvider(), name=tag, transport=http_obj)
    log.DeletedResource('[{tag}] (referencing [{digest}])'.format(tag=tag, digest=digests[tag]))