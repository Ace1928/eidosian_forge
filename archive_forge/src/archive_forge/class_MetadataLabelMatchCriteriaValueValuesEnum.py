from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MetadataLabelMatchCriteriaValueValuesEnum(_messages.Enum):
    """Specifies how matching should be done. Supported values are:
    MATCH_ANY: At least one of the Labels specified in the matcher should
    match the metadata presented by xDS client. MATCH_ALL: The metadata
    presented by the xDS client should contain all of the labels specified
    here. The selection is determined based on the best match. For example,
    suppose there are three EndpointPolicy resources P1, P2 and P3 and if P1
    has a the matcher as MATCH_ANY , P2 has MATCH_ALL , and P3 has MATCH_ALL .
    If a client with label connects, the config from P1 will be selected. If a
    client with label connects, the config from P2 will be selected. If a
    client with label connects, the config from P3 will be selected. If there
    is more than one best match, (for example, if a config P4 with selector
    exists and if a client with label connects), pick up the one with older
    creation time.

    Values:
      METADATA_LABEL_MATCH_CRITERIA_UNSPECIFIED: Default value. Should not be
        used.
      MATCH_ANY: At least one of the Labels specified in the matcher should
        match the metadata presented by xDS client.
      MATCH_ALL: The metadata presented by the xDS client should contain all
        of the labels specified here.
    """
    METADATA_LABEL_MATCH_CRITERIA_UNSPECIFIED = 0
    MATCH_ANY = 1
    MATCH_ALL = 2