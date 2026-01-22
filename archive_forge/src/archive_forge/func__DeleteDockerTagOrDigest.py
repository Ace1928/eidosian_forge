from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_session
from googlecloudsdk.api_lib.container.images import util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
import six
def _DeleteDockerTagOrDigest(self, tag_or_digest, http_obj):
    docker_session.Delete(creds=util.CredentialProvider(), name=tag_or_digest, transport=http_obj)
    log.DeletedResource(tag_or_digest)