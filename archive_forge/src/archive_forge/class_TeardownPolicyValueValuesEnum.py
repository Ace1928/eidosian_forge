from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TeardownPolicyValueValuesEnum(_messages.Enum):
    """Sets the policy for determining when to turndown worker pool. Allowed
    values are: `TEARDOWN_ALWAYS`, `TEARDOWN_ON_SUCCESS`, and
    `TEARDOWN_NEVER`. `TEARDOWN_ALWAYS` means workers are always torn down
    regardless of whether the job succeeds. `TEARDOWN_ON_SUCCESS` means
    workers are torn down if the job succeeds. `TEARDOWN_NEVER` means the
    workers are never torn down. If the workers are not torn down by the
    service, they will continue to run and use Google Compute Engine VM
    resources in the user's project until they are explicitly terminated by
    the user. Because of this, Google recommends using the `TEARDOWN_ALWAYS`
    policy except for small, manually supervised test jobs. If unknown or
    unspecified, the service will attempt to choose a reasonable default.

    Values:
      TEARDOWN_POLICY_UNKNOWN: The teardown policy isn't specified, or is
        unknown.
      TEARDOWN_ALWAYS: Always teardown the resource.
      TEARDOWN_ON_SUCCESS: Teardown the resource on success. This is useful
        for debugging failures.
      TEARDOWN_NEVER: Never teardown the resource. This is useful for
        debugging and development.
    """
    TEARDOWN_POLICY_UNKNOWN = 0
    TEARDOWN_ALWAYS = 1
    TEARDOWN_ON_SUCCESS = 2
    TEARDOWN_NEVER = 3