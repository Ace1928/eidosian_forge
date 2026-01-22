from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import io
import sys
import threading
import time
from apitools.base.py import encoding_helper
from apitools.base.py.exceptions import HttpConflictError
from apitools.base.py.exceptions import HttpError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import iap_tunnel
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import exceptions as tpu_exceptions
from googlecloudsdk.command_lib.compute.tpus.tpu_vm import util as tpu_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util.files import FileWriter
import six
def AddTPUSSHArgs(parser, enable_iap, enable_batching=False, enable_batching_default='all'):
    """Arguments that are common and specific to both TPU VM/QR SSH and SCP."""
    parser.add_argument('--worker', default='0', help='          TPU worker to connect to. The supported value is a single 0-based\n          index of the worker in the case of a TPU Pod. When also using the\n          `--command` flag, it additionally supports a comma-separated list\n          (e.g. \'1,4,6\'), range (e.g. \'1-3\'), or special keyword ``all" to\n          run the command concurrently on each of the specified workers.\n\n          Note that when targeting multiple workers, you should run \'ssh-add\'\n          with your private key prior to executing the gcloud command. Default:\n          \'ssh-add ~/.ssh/google_compute_engine\'.\n          ')
    if enable_batching:
        parser.add_argument('--batch-size', default=enable_batching_default, help="            Batch size for simultaneous command execution on the client's side.\n            When using a comma-separated list (e.g. '1,4,6') or a range (e.g. '1-3') or\n            ``all`` keyword in `--worker` flag, it executes the command\n            concurrently in groups of the batch size. This flag takes a\n            value greater than 0 to specify the batch size to control the\n            concurrent connections that can be established with the TPU\n            workers, or the special keyword ``all`` to allow the concurrent\n            command executions on all the specified workers in `--worker` flag.\n            Maximum value of this flag should not be more than the number of\n            specified workers, otherwise the value will be treated as\n            ``--batch-size=all``.\n            ")
    if enable_iap:
        routing_group = parser.add_mutually_exclusive_group()
        routing_group.add_argument('--internal-ip', action='store_true', help='            Connect to TPU VMs using their internal IP addresses rather than their\n            external IP addresses. Use this to connect from a Google Compute\n            Engine VM to a TPU VM on the same VPC network, or between two peered\n            VPC networks.\n            ')
        routing_group.add_argument('--tunnel-through-iap', action='store_true', help='        Tunnel the SSH connection through Cloud Identity-Aware Proxy for TCP\n        forwarding.\n\n        This flag must be specified to attempt to connect via IAP tunneling. If it\n        is not set, and connection to a Cloud TPU VM without external IP address\n        is attempted from outside the network, then the command will fail.\n\n        To use IAP tunneling, there must be firewall access to the SSH port for\n        the IAP TCP IP address range for the network the TPU is created in. See\n        the [user guide](https://cloud.google.com/iap/docs/using-tcp-forwarding)\n        for more details.\n\n        To learn more, see the\n        [IAP for TCP forwarding documentation](https://cloud.google.com/iap/docs/tcp-forwarding-overview).\n        ')
    else:
        parser.add_argument('--internal-ip', action='store_true', help='            Connect to TPU VMs using their internal IP addresses rather than their\n            external IP addresses. Use this to connect from a Google Compute\n            Engine VM to a TPU VM on the same VPC network, or between two peered\n            VPC networks.\n            ')