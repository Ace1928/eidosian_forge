from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os.path
import threading
import time
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.queued_resources import ssh as qr_ssh_utils
from googlecloudsdk.command_lib.compute.tpus.queued_resources import util as queued_resource_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import ssh as tpu_ssh_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def AddSSHArgs(parser):
    """Additional flags and positional args to be passed to *ssh(1)*."""
    parser.add_argument('--ssh-flag', action='append', help='      Additional flags to be passed to *ssh(1)*. It is recommended that flags\n      be passed using an assignment operator and quotes. Example:\n\n        $ {command} example-instance --zone=us-central1-a --ssh-flag="-vvv" --ssh-flag="-L 80:localhost:80"\n\n      This flag will replace occurences of ``%USER%\'\' and ``%TPU%\'\' with\n      their dereferenced values. For example, passing ``80:%TPU%:80`` into\n      the flag is equivalent to passing ``80:162.222.181.197:80\'\' to *ssh(1)*\n      if the external IP address of \'example-instance\' is 162.222.181.197.\n\n      If connecting to the instance\'s external IP, then %TPU% is replaced\n      with that, otherwise it is replaced with the internal IP.\n      ')
    parser.add_argument('user_queued_resource', completer=completers.InstancesCompleter, metavar='[USER@]QR', help="      Specifies the Cloud TPU Queued Resource to send SSH command to.\n\n      ``USER'' specifies the username with which to SSH. If omitted, the user\n      login name is used.\n\n      ``QR'' specifies the name of the Cloud TPU Queued Resource to send SSH command to.\n      ")
    parser.add_argument('ssh_args', nargs=argparse.REMAINDER, help='          Flags and positionals passed to the underlying ssh implementation.\n          ', example='        $ {command} example-instance --zone=us-central1-a -- -vvv -L 80:%TPU%:80\n      ')
    parser.add_argument('--node', default='0', help='          TPU node(s) to connect to. The supported value is a single 0-based\n          index of the node(s) in the case of a TPU Pod. When also using the\n          `--command` flag, it additionally supports a comma-separated list\n          (e.g. \'1,4,6\'), range (e.g. \'1-3\'), or special keyword ``all" to\n          run the command concurrently on each of the specified node(s).\n\n          Note that when targeting multiple nodes, you should run \'ssh-add\'\n          with your private key prior to executing the gcloud command. Default:\n          \'ssh-add ~/.ssh/google_compute_engine\'.\n          ')