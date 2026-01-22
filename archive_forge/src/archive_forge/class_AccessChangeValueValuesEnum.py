from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessChangeValueValuesEnum(_messages.Enum):
    """How the principal's access, specified in the AccessState field,
    changed between the current (baseline) policies and proposed (simulated)
    policies.

    Values:
      ACCESS_CHANGE_TYPE_UNSPECIFIED: Default value. This value is unused.
      NO_CHANGE: The principal's access did not change. This includes the case
        where both baseline and simulated are UNKNOWN, but the unknown
        information is equivalent.
      UNKNOWN_CHANGE: The principal's access under both the current policies
        and the proposed policies is `UNKNOWN`, but the unknown information
        differs between them.
      ACCESS_REVOKED: The principal had access under the current policies
        (`GRANTED`), but will no longer have access after the proposed changes
        (`NOT_GRANTED`).
      ACCESS_GAINED: The principal did not have access under the current
        policies (`NOT_GRANTED`), but will have access after the proposed
        changes (`GRANTED`).
      ACCESS_MAYBE_REVOKED: This result can occur for the following reasons: *
        The principal had access under the current policies (`GRANTED`), but
        their access after the proposed changes is `UNKNOWN`. * The
        principal's access under the current policies is `UNKNOWN`, but they
        will not have access after the proposed changes (`NOT_GRANTED`).
      ACCESS_MAYBE_GAINED: This result can occur for the following reasons: *
        The principal did not have access under the current policies
        (`NOT_GRANTED`), but their access after the proposed changes is
        `UNKNOWN`. * The principal's access under the current policies is
        `UNKNOWN`, but they will have access after the proposed changes
        (`GRANTED`).
    """
    ACCESS_CHANGE_TYPE_UNSPECIFIED = 0
    NO_CHANGE = 1
    UNKNOWN_CHANGE = 2
    ACCESS_REVOKED = 3
    ACCESS_GAINED = 4
    ACCESS_MAYBE_REVOKED = 5
    ACCESS_MAYBE_GAINED = 6