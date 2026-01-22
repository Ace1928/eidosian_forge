from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DeployFailureCauseValueValuesEnum(_messages.Enum):
    """Output only. The reason this rollout failed. This will always be
    unspecified while the rollout is in progress.

    Values:
      FAILURE_CAUSE_UNSPECIFIED: No reason for failure is specified.
      CLOUD_BUILD_UNAVAILABLE: Cloud Build is not available, either because it
        is not enabled or because Cloud Deploy has insufficient permissions.
        See [required permission](https://cloud.google.com/deploy/docs/cloud-
        deploy-service-account#required_permissions).
      EXECUTION_FAILED: The deploy operation did not complete successfully;
        check Cloud Build logs.
      DEADLINE_EXCEEDED: Deployment did not complete within the alloted time.
      RELEASE_FAILED: Release is in a failed state.
      RELEASE_ABANDONED: Release is abandoned.
      VERIFICATION_CONFIG_NOT_FOUND: No Skaffold verify configuration was
        found.
      CLOUD_BUILD_REQUEST_FAILED: Cloud Build failed to fulfill Cloud Deploy's
        request. See failure_message for additional details.
      OPERATION_FEATURE_NOT_SUPPORTED: A Rollout operation had a feature
        configured that is not supported.
    """
    FAILURE_CAUSE_UNSPECIFIED = 0
    CLOUD_BUILD_UNAVAILABLE = 1
    EXECUTION_FAILED = 2
    DEADLINE_EXCEEDED = 3
    RELEASE_FAILED = 4
    RELEASE_ABANDONED = 5
    VERIFICATION_CONFIG_NOT_FOUND = 6
    CLOUD_BUILD_REQUEST_FAILED = 7
    OPERATION_FEATURE_NOT_SUPPORTED = 8