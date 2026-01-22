from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AnthoseventsNamespacesTriggersDeleteRequest(_messages.Message):
    """A AnthoseventsNamespacesTriggersDeleteRequest object.

  Fields:
    apiVersion: Specifies the target version for the cascading deletion
      policy. Events for Cloud Run currently ignores this parameter.
    kind: Specifies the cascading deletion policy, in conjunction of
      `progation_policy` field. Events for Cloud Run currently ignores this
      parameter.
    name: The name of the trigger being deleted. If needed, replace
      {namespace_id} with the project ID.
    propagationPolicy: Specifies the propagation policy of delete. Events for
      Cloud Run currently ignores this setting, and deletes in the background.
      Please see kubernetes.io/docs/concepts/workloads/controllers/garbage-
      collection/ for more information.
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    name = _messages.StringField(3, required=True)
    propagationPolicy = _messages.StringField(4)