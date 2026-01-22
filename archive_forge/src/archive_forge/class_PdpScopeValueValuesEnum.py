from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PdpScopeValueValuesEnum(_messages.Enum):
    """Specifies how child public delegated prefix will be scoped. It could
    be one of following values: - `REGIONAL`: The public delegated prefix is
    regional only. The provisioning will take a few minutes. - `GLOBAL`: The
    public delegated prefix is global only. The provisioning will take ~4
    weeks. - `GLOBAL_AND_REGIONAL` [output only]: The public delegated
    prefixes is BYOIP V1 legacy prefix. This is output only value and no
    longer supported in BYOIP V2.

    Values:
      GLOBAL: The public delegated prefix is global only. The provisioning
        will take ~4 weeks.
      GLOBAL_AND_REGIONAL: The public delegated prefixes is BYOIP V1 legacy
        prefix. This is output only value and no longer supported in BYOIP V2.
      REGIONAL: The public delegated prefix is regional only. The provisioning
        will take a few minutes.
    """
    GLOBAL = 0
    GLOBAL_AND_REGIONAL = 1
    REGIONAL = 2