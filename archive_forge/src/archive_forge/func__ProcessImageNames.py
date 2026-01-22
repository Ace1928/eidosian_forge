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
def _ProcessImageNames(self, image_names):
    digests = set()
    tags = set()
    for image_name in image_names:
        docker_obj = util.GetDockerImageFromTagOrDigest(image_name)
        if isinstance(docker_obj, docker_name.Digest):
            digests.add(docker_obj)
        elif isinstance(docker_obj, docker_name.Tag):
            if not util.IsFullySpecified(image_name):
                log.warning('Implicit ":latest" tag specified: ' + image_name)
            tags.add(docker_obj)
    return [digests, tags]