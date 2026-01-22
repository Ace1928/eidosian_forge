from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class GoogleCloudOrgpolicyV2AlternatePolicySpec(_messages.Message):
    """Similar to PolicySpec but with an extra 'launch' field for launch
  reference. The PolicySpec here is specific for dry-run/darklaunch.

  Fields:
    launch: Reference to the launch that will be used while audit logging and
      to control the launch. Should be set only in the alternate policy.
    spec: Specify constraint for configurations of Google Cloud resources.
  """
    launch = _messages.StringField(1)
    spec = _messages.MessageField('GoogleCloudOrgpolicyV2PolicySpec', 2)