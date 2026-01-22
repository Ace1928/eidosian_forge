from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmDnsSettingValueValuesEnum(_messages.Enum):
    """[Output Only] Default internal DNS setting used by VMs running in this
    project.

    Values:
      GLOBAL_DEFAULT: <no description>
      UNSPECIFIED_VM_DNS_SETTING: <no description>
      ZONAL_DEFAULT: <no description>
      ZONAL_ONLY: <no description>
    """
    GLOBAL_DEFAULT = 0
    UNSPECIFIED_VM_DNS_SETTING = 1
    ZONAL_DEFAULT = 2
    ZONAL_ONLY = 3