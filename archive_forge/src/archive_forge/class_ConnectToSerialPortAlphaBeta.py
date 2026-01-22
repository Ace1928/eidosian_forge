from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import http
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from six.moves import http_client as httplib
@base.ReleaseTracks(base.ReleaseTrack.BETA, base.ReleaseTrack.ALPHA)
class ConnectToSerialPortAlphaBeta(ConnectToSerialPort):
    """Connect to the serial port of an instance.

  *{command}* allows users to connect to, and interact with, a VM's
  virtual serial port using ssh as the secure, authenticated transport
  protocol.

  The user must first enable serial port access to a given VM by setting
  the 'serial-port-enable=true' metadata key-value pair. Setting
  'serial-port-enable' on the project-level metadata enables serial port
  access to all VMs in the project.

  This command uses the same SSH key pair as the `gcloud compute ssh`
  command and also ensures that the user's public SSH key is present in
  the project's metadata. If the user does not have a public SSH key,
  one is generated using ssh-keygen.

  ## EXAMPLES
  To connect to the serial port of the instance 'my-instance' in zone
  'us-central1-f', run:

    $ {command} my-instance --zone=us-central1-f
  """

    @classmethod
    def Args(cls, parser):
        super(ConnectToSerialPortAlphaBeta, cls).Args(parser)