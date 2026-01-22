from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartnerValueValuesEnum(_messages.Enum):
    """Optional. Partner regime associated with this workload.

    Values:
      PARTNER_UNSPECIFIED: <no description>
      LOCAL_CONTROLS_BY_S3NS: Enum representing S3NS (Thales) partner.
      SOVEREIGN_CONTROLS_BY_T_SYSTEMS: Enum representing T_SYSTEM (TSI)
        partner.
      SOVEREIGN_CONTROLS_BY_SIA_MINSAIT: Enum representing SIA_MINSAIT (Indra)
        partner.
      SOVEREIGN_CONTROLS_BY_PSN: Enum representing PSN (TIM) partner.
    """
    PARTNER_UNSPECIFIED = 0
    LOCAL_CONTROLS_BY_S3NS = 1
    SOVEREIGN_CONTROLS_BY_T_SYSTEMS = 2
    SOVEREIGN_CONTROLS_BY_SIA_MINSAIT = 3
    SOVEREIGN_CONTROLS_BY_PSN = 4