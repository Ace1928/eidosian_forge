from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Node(_messages.Message):
    """Kubernetes nodes associated with the finding.

  Fields:
    name: [Full resource name](https://google.aip.dev/122#full-resource-names)
      of the Compute Engine VM running the cluster node.
  """
    name = _messages.StringField(1)