from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerStatusStateful(_messages.Message):
    """A InstanceGroupManagerStatusStateful object.

  Fields:
    hasStatefulConfig: [Output Only] A bit indicating whether the managed
      instance group has stateful configuration, that is, if you have
      configured any items in a stateful policy or in per-instance configs.
      The group might report that it has no stateful configuration even when
      there is still some preserved state on a managed instance, for example,
      if you have deleted all PICs but not yet applied those deletions.
    isStateful: [Output Only] A bit indicating whether the managed instance
      group has stateful configuration, that is, if you have configured any
      items in a stateful policy or in per-instance configs. The group might
      report that it has no stateful configuration even when there is still
      some preserved state on a managed instance, for example, if you have
      deleted all PICs but not yet applied those deletions. This field is
      deprecated in favor of has_stateful_config.
    perInstanceConfigs: [Output Only] Status of per-instance configurations on
      the instances.
  """
    hasStatefulConfig = _messages.BooleanField(1)
    isStateful = _messages.BooleanField(2)
    perInstanceConfigs = _messages.MessageField('InstanceGroupManagerStatusStatefulPerInstanceConfigs', 3)