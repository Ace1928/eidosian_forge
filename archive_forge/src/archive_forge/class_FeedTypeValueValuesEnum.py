from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeedTypeValueValuesEnum(_messages.Enum):
    """Required. Type feed to be ingested into condor

    Values:
      FEEDTYPE_UNSPECIFIED: <no description>
      RESOURCE_METADATA: Database resource metadata feed from control plane
      OBSERVABILITY_DATA: Database resource monitoring data
      SECURITY_FINDING_DATA: Database resource security health signal data
      RECOMMENDATION_SIGNAL_DATA: Database resource recommendation signal data
    """
    FEEDTYPE_UNSPECIFIED = 0
    RESOURCE_METADATA = 1
    OBSERVABILITY_DATA = 2
    SECURITY_FINDING_DATA = 3
    RECOMMENDATION_SIGNAL_DATA = 4