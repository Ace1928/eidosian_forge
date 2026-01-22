from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RegionInstanceGroupManagerPatchInstanceConfigReq(_messages.Message):
    """RegionInstanceGroupManagers.patchPerInstanceConfigs

  Fields:
    perInstanceConfigs: The list of per-instance configurations to insert or
      patch on this managed instance group.
  """
    perInstanceConfigs = _messages.MessageField('PerInstanceConfig', 1, repeated=True)