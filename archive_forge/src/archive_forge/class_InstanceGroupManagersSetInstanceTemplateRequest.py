from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagersSetInstanceTemplateRequest(_messages.Message):
    """A InstanceGroupManagersSetInstanceTemplateRequest object.

  Fields:
    instanceTemplate: The URL of the instance template that is specified for
      this managed instance group. The group uses this template to create all
      new instances in the managed instance group. The templates for existing
      instances in the group do not change unless you run recreateInstances,
      run applyUpdatesToInstances, or set the group's updatePolicy.type to
      PROACTIVE.
  """
    instanceTemplate = _messages.StringField(1)