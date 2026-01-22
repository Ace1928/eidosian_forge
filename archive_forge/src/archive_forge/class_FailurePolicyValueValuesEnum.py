from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FailurePolicyValueValuesEnum(_messages.Enum):
    """Represents the failure policy of a pipeline. Currently, the default of
    a pipeline is that the pipeline will continue to run until no more tasks
    can be executed, also known as PIPELINE_FAILURE_POLICY_FAIL_SLOW. However,
    if a pipeline is set to PIPELINE_FAILURE_POLICY_FAIL_FAST, it will stop
    scheduling any new tasks when a task has failed. Any scheduled tasks will
    continue to completion.

    Values:
      PIPELINE_FAILURE_POLICY_UNSPECIFIED: Default value, and follows fail
        slow behavior.
      PIPELINE_FAILURE_POLICY_FAIL_SLOW: Indicates that the pipeline should
        continue to run until all possible tasks have been scheduled and
        completed.
      PIPELINE_FAILURE_POLICY_FAIL_FAST: Indicates that the pipeline should
        stop scheduling new tasks after a task has failed.
    """
    PIPELINE_FAILURE_POLICY_UNSPECIFIED = 0
    PIPELINE_FAILURE_POLICY_FAIL_SLOW = 1
    PIPELINE_FAILURE_POLICY_FAIL_FAST = 2