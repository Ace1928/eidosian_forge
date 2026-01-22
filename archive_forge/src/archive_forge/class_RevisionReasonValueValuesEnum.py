from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RevisionReasonValueValuesEnum(_messages.Enum):
    """Output only. A reason for the revision condition.

    Values:
      REVISION_REASON_UNDEFINED: Default value.
      PENDING: Revision in Pending state.
      RESERVE: Revision is in Reserve state.
      RETIRED: Revision is Retired.
      RETIRING: Revision is being retired.
      RECREATING: Revision is being recreated.
      HEALTH_CHECK_CONTAINER_ERROR: There was a health check error.
      CUSTOMIZED_PATH_RESPONSE_PENDING: Health check failed due to user error
        from customized path of the container. System will retry.
      MIN_INSTANCES_NOT_PROVISIONED: A revision with min_instance_count > 0
        was created and is reserved, but it was not configured to serve
        traffic, so it's not live. This can also happen momentarily during
        traffic migration.
      ACTIVE_REVISION_LIMIT_REACHED: The maximum allowed number of active
        revisions has been reached.
      NO_DEPLOYMENT: There was no deployment defined. This value is no longer
        used, but Services created in older versions of the API might contain
        this value.
      HEALTH_CHECK_SKIPPED: A revision's container has no port specified since
        the revision is of a manually scaled service with 0 instance count
      MIN_INSTANCES_WARMING: A revision with min_instance_count > 0 was
        created and is waiting for enough instances to begin a traffic
        migration.
    """
    REVISION_REASON_UNDEFINED = 0
    PENDING = 1
    RESERVE = 2
    RETIRED = 3
    RETIRING = 4
    RECREATING = 5
    HEALTH_CHECK_CONTAINER_ERROR = 6
    CUSTOMIZED_PATH_RESPONSE_PENDING = 7
    MIN_INSTANCES_NOT_PROVISIONED = 8
    ACTIVE_REVISION_LIMIT_REACHED = 9
    NO_DEPLOYMENT = 10
    HEALTH_CHECK_SKIPPED = 11
    MIN_INSTANCES_WARMING = 12