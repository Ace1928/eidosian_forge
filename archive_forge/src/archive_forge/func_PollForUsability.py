from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import time
from googlecloudsdk.api_lib.services import enable_api
from googlecloudsdk.command_lib.container.fleet import resources
from googlecloudsdk.command_lib.container.fleet.features import base
from googlecloudsdk.command_lib.container.fleet.features import info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import retry
def PollForUsability(self):
    message = 'Waiting for controller to start...'
    aborted_message = 'Aborting wait for controller to start.\n'
    timeout = 120000
    timeout_message = 'Please use the `describe` command to check Featurestate for debugging information.\n'
    ok_code = self.messages.FeatureState.CodeValueValuesEnum.OK
    try:
        with progress_tracker.ProgressTracker(message, aborted_message=aborted_message) as tracker:
            time.sleep(5)

            def _StatusUpdate(unused_result, unused_status):
                tracker.Tick()
            retryer = retry.Retryer(max_wait_ms=timeout, wait_ceiling_ms=1000, status_update_func=_StatusUpdate)

            def _PollFunc():
                return self.GetFeature()

            def _IsNotDone(feature, unused_state):
                if feature.state is None or feature.state.state is None:
                    return True
                return feature.state.state.code != ok_code
            return retryer.RetryOnResult(func=_PollFunc, should_retry_if=_IsNotDone, sleep_ms=500)
    except retry.WaitException:
        raise exceptions.Error('Controller did not start in {} minutes. {}'.format(timeout / 60000, timeout_message))