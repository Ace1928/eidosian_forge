from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExternalManagedMigrationStateValueValuesEnum(_messages.Enum):
    """Specifies the canary migration state. Possible values are PREPARE,
    TEST, and FINALIZE. To begin the migration from EXTERNAL to
    EXTERNAL_MANAGED, the state must be changed to PREPARE. The state must be
    changed to FINALIZE before the loadBalancingScheme can be changed to
    EXTERNAL_MANAGED. Optionally, the TEST state can be used to migrate
    traffic by percentage using externalManagedMigrationTestingPercentage.
    Rolling back a migration requires the states to be set in reverse order.
    So changing the scheme from EXTERNAL_MANAGED to EXTERNAL requires the
    state to be set to FINALIZE at the same time. Optionally, the TEST state
    can be used to migrate some traffic back to EXTERNAL or PREPARE can be
    used to migrate all traffic back to EXTERNAL.

    Values:
      FINALIZE: <no description>
      PREPARE: <no description>
      TEST: <no description>
    """
    FINALIZE = 0
    PREPARE = 1
    TEST = 2