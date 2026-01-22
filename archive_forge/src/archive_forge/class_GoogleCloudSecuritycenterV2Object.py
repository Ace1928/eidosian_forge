from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2Object(_messages.Message):
    """Kubernetes object related to the finding, uniquely identified by GKNN.
  Used if the object Kind is not one of Pod, Node, NodePool, Binding, or
  AccessReview.

  Fields:
    containers: Pod containers associated with this finding, if any.
    group: Kubernetes object group, such as "policy.k8s.io/v1".
    kind: Kubernetes object kind, such as "Namespace".
    name: Kubernetes object name. For details see
      https://kubernetes.io/docs/concepts/overview/working-with-
      objects/names/.
    ns: Kubernetes object namespace. Must be a valid DNS label. Named "ns" to
      avoid collision with C++ namespace keyword. For details see
      https://kubernetes.io/docs/tasks/administer-cluster/namespaces/.
  """
    containers = _messages.MessageField('GoogleCloudSecuritycenterV2Container', 1, repeated=True)
    group = _messages.StringField(2)
    kind = _messages.StringField(3)
    name = _messages.StringField(4)
    ns = _messages.StringField(5)