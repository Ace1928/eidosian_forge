from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.command_lib.compute.instance_groups.managed import wait_info
from googlecloudsdk.core import log
from googlecloudsdk.core.util import retry
def WaitForIgmState(client, group_ref, desired_igm_state, timeout_sec=None):
    """Waits until the desired state of managed instance group is reached."""
    try:
        max_wait_ms = timeout_sec * 1000 if timeout_sec else None
        retry.Retryer(max_wait_ms=max_wait_ms).RetryOnResult(_IsStateReached, [client, group_ref, desired_igm_state], should_retry_if=False, sleep_ms=_TIME_BETWEEN_POLLS_MS)
        if desired_igm_state == IgmState.STABLE:
            log.out.Print('Group is stable')
        elif desired_igm_state == IgmState.VERSION_TARGET_REACHED:
            log.out.Print('Version target is reached')
        elif desired_igm_state == IgmState.ALL_INSTANCES_CONFIG_EFFECTIVE:
            log.out.Print('All-instances config is effective')
    except retry.WaitException:
        if desired_igm_state == IgmState.STABLE:
            raise utils.TimeoutError('Timeout while waiting for group to become stable.')
        if desired_igm_state == IgmState.VERSION_TARGET_REACHED:
            raise utils.TimeoutError('Timeout while waiting for group to reach version target.')
        if desired_igm_state == IgmState.ALL_INSTANCES_CONFIG_EFFECTIVE:
            raise utils.TimeoutError('Timeout while waiting for group to reach effective all-instances config.')