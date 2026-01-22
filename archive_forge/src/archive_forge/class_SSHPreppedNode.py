from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
class SSHPreppedNode(object):
    """Object that has all the data needed to successfully SSH into a node.

  Attributes:
    worker_ips: The IPs of the workers of the node.
    ssh_helper: The ssh_helper used to SSH into the node.
    id: The id of the node.
    tpu_name: The unqualified TPU VM name.
    instance_names: The name of the instances of the workers of the node.
    project: The project associated with the node.
    command_list: The list of the commands passed into ssh.
    remainder: The remainder list of ssh_args used to pass into the SSH command.
    host_key_suffixes: The host key suffixes associated with the node.
    user: The user executing the SSH command.
    release_track: The release track for the SSH protos (Alpha, Beta, etc.).
    enable_batching: A bool indicating if the user enabled batching for the
      node.
  """

    def __init__(self, tpu_name, user, release_track, enable_batching):
        self.tpu_name = tpu_name
        self.user = user
        self.release_track = release_track
        self.enable_batching = enable_batching
        self.worker_ips = []
        self.ssh_helper = None
        self.id = None
        self.instance_names = []
        self.project = None
        self.command_list = []
        self.remainder = None
        self.host_key_suffixes = []