from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from googlecloudsdk.command_lib.deploy import automation_util
from googlecloudsdk.command_lib.deploy import deploy_util
from googlecloudsdk.command_lib.deploy import exceptions
from googlecloudsdk.command_lib.deploy import target_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ParseRepairMode(messages, modes):
    """Parses RepairMode of the Automation resource."""
    modes_pb = []
    for m in modes:
        mode = messages.RepairMode()
        if RETRY_FIELD in m:
            mode.retry = messages.Retry()
            retry = m.get(RETRY_FIELD)
            if retry:
                mode.retry.attempts = retry.get(ATTEMPTS_FIELD)
                mode.retry.wait = _WaitMinToSec(retry.get(WAIT_FIELD))
                mode.retry.backoffMode = _ParseBackoffMode(messages, retry.get(BACKOFF_MODE_FIELD))
        if ROLLBACK_FIELD in m:
            mode.rollback = messages.Rollback()
            rollback = m.get(ROLLBACK_FIELD)
            if rollback:
                mode.rollback.destinationPhase = rollback.get(DESTINATION_PHASE_FIELD)
        modes_pb.append(mode)
    return modes_pb