from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import log
import six
def WaitForNodeBatchCompletion(ssh_threads, nodes):
    """Waits for the completion of batch, but does not block for failures.

  Args:
    ssh_threads: List of ssh threads.
    nodes: List of SSH prepped nodes.
  """
    for ssh_thread in ssh_threads:
        ssh_thread.join()
    for node in nodes:
        if node:
            log.status.Print('Finished preparing node {}.'.format(node.tpu_name))