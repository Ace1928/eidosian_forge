from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import threading
import time
from argcomplete.completers import FilesCompleter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.queued_resources import ssh as qr_ssh_utils
from googlecloudsdk.command_lib.compute.tpus.queued_resources import util as queued_resource_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import ssh as tpu_ssh_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def AddSCPArgs(parser):
    """Additional flags and positional args to be passed to *scp(1)*."""
    parser.add_argument('--scp-flag', action='append', help='      Additional flags to be passed to *scp(1)*. This flag may be repeated.\n      ')
    parser.add_argument('sources', completer=FilesCompleter, help='Specifies the files to copy.', metavar='[[USER@]INSTANCE:]SRC', nargs='+')
    parser.add_argument('destination', help='Specifies a destination for the source files.', metavar='[[USER@]INSTANCE:]DEST')
    parser.add_argument('--recurse', action='store_true', help='Upload directories recursively.')
    parser.add_argument('--compress', action='store_true', help='Enable compression.')
    parser.add_argument('--node', default='0', help='          TPU node(s) to connect to. The supported value is a single 0-based\n          index of the node(s) in the case of a TPU Pod. It additionally\n          supports a comma-separated list (e.g. \'1,4,6\'), range (e.g. \'1-3\'), or\n          special keyword ``all" to run the command concurrently on each of the\n          specified node(s).\n\n          Note that when targeting multiple nodes, you should run \'ssh-add\'\n          with your private key prior to executing the gcloud command. Default:\n          \'ssh-add ~/.ssh/google_compute_engine\'.\n          ')