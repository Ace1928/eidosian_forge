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
def GetHostKeySuffixesFromGuestAttributes(guest_attributes_response, single_pod_worker, worker_ips, node):
    """Retrieves the host key suffixes from the GuestAttributes."""
    if single_pod_worker:
        worker_id = list(worker_ips)[0]
        return _ParseSingleHostKeySuffix(guest_attributes_response.guestAttributes, len(node.networkEndpoints), worker_id)
    else:
        return _ParseHostKeySuffixes(guest_attributes_response.guestAttributes)