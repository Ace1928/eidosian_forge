from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckerTypeValueValuesEnum(_messages.Enum):
    """The type of checkers to use to execute the Uptime check.

    Values:
      CHECKER_TYPE_UNSPECIFIED: The default checker type. Currently converted
        to STATIC_IP_CHECKERS on creation, the default conversion behavior may
        change in the future.
      STATIC_IP_CHECKERS: STATIC_IP_CHECKERS are used for uptime checks that
        perform egress across the public internet. STATIC_IP_CHECKERS use the
        static IP addresses returned by ListUptimeCheckIps.
      VPC_CHECKERS: VPC_CHECKERS are used for uptime checks that perform
        egress using Service Directory and private network access. When using
        VPC_CHECKERS, the monitored resource type must be
        servicedirectory_service.
    """
    CHECKER_TYPE_UNSPECIFIED = 0
    STATIC_IP_CHECKERS = 1
    VPC_CHECKERS = 2