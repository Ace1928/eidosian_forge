from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuorumTypeValueValuesEnum(_messages.Enum):
    """Output only. The `QuorumType` of the instance configuration.

    Values:
      QUORUM_TYPE_UNSPECIFIED: Not specified.
      REGION: An instance configuration tagged with REGION quorum type forms a
        write quorum in a single region.
      DUAL_REGION: An instance configuration tagged with DUAL_REGION quorum
        type forms a write quorums with exactly two read-write regions in a
        multi-region configuration. This instance configurations requires
        reconfiguration in the event of regional failures.
      MULTI_REGION: An instance configuration tagged with MULTI_REGION quorum
        type forms a write quorums from replicas are spread across more than
        one region in a multi-region configuration.
    """
    QUORUM_TYPE_UNSPECIFIED = 0
    REGION = 1
    DUAL_REGION = 2
    MULTI_REGION = 3