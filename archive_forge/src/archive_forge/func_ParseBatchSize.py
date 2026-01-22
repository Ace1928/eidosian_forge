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
def ParseBatchSize(batch_size_flag, num_ips):
    """Parses the --batch-size flag and validates the flag value.

  Args:
    batch_size_flag: str, batch-size flag argument.
    num_ips: int, number of ip-addresses for the ssh command execution.

  Returns:
    int, batch-size value capped at number of workers in num_ips.

  Raises:
    InvalidArgumentException, if the batch_size_flag is neither a positive
    integer nor equal to the `all` keyword.
  """
    if six.text_type(batch_size_flag).upper() == 'ALL':
        if num_ips > 100:
            log.warning('Executing ssh command on too many workers simultaneously is prone to error. Command may fail. Please consider using `--batch-size` flag if the command fails, for example, --batch-size=100.')
        return num_ips
    else:
        try:
            if int(batch_size_flag) > 0:
                return min(int(batch_size_flag), num_ips)
            else:
                raise ValueError()
        except ValueError as error:
            six.raise_from(exceptions.InvalidArgumentException('--batch-size', 'unable to parse the batch size value {}. Please use a positive integer not more than the number of TPU workers.'.format(batch_size_flag)), error)