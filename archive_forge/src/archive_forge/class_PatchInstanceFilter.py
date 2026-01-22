from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PatchInstanceFilter(_messages.Message):
    """A filter to target VM instances for patching. The targeted VMs must meet
  all criteria specified. So if both labels and zones are specified, the patch
  job targets only VMs with those labels and in those zones.

  Fields:
    all: Target all VM instances in the project. If true, no other criteria is
      permitted.
    groupLabels: Targets VM instances matching at least one of these label
      sets. This allows targeting of disparate groups, for example "env=prod
      or env=staging".
    instanceNamePrefixes: Targets VMs whose name starts with one of these
      prefixes. Similar to labels, this is another way to group VMs when
      targeting configs, for example prefix="prod-".
    instances: Targets any of the VM instances specified. Instances are
      specified by their URI in the form
      `zones/[ZONE]/instances/[INSTANCE_NAME]`,
      `projects/[PROJECT_ID]/zones/[ZONE]/instances/[INSTANCE_NAME]`, or `http
      s://www.googleapis.com/compute/v1/projects/[PROJECT_ID]/zones/[ZONE]/ins
      tances/[INSTANCE_NAME]`
    zones: Targets VM instances in ANY of these zones. Leave empty to target
      VM instances in any zone.
  """
    all = _messages.BooleanField(1)
    groupLabels = _messages.MessageField('PatchInstanceFilterGroupLabel', 2, repeated=True)
    instanceNamePrefixes = _messages.StringField(3, repeated=True)
    instances = _messages.StringField(4, repeated=True)
    zones = _messages.StringField(5, repeated=True)