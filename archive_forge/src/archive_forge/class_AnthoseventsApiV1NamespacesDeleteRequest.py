from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsApiV1NamespacesDeleteRequest(_messages.Message):
    """A AnthoseventsApiV1NamespacesDeleteRequest object.

  Fields:
    apiVersion: Cloud Run currently ignores this parameter.
    kind: Cloud Run currently ignores this parameter.
    name: Required. The name of the namespace being deleted. If needed,
      replace {namespace_id} with the project ID.
    propagationPolicy: Specifies the propagation policy of delete. Cloud Run
      currently ignores this setting, and deletes in the background. Please
      see kubernetes.io/docs/concepts/workloads/controllers/garbage-
      collection/ for more information.
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    propagationPolicy = _messages.StringField(4)