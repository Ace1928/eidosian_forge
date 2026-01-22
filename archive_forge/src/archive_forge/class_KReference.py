from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class KReference(_messages.Message):
    """from
  https://github.com/knative/pkg/blob/master/apis/duck/v1/knative_reference.go
  KReference contains enough information to refer to another object. It's a
  trimmed down version of corev1.ObjectReference.

  Fields:
    apiVersion: API version of the referent.
    kind: Kind of the referent. More info:
      https://git.k8s.io/community/contributors/devel/sig-architecture/api-
      conventions.md#types-kinds
    name: Name of the referent. More info:
      https://kubernetes.io/docs/concepts/overview/working-with-
      objects/names/#names
    namespace: Namespace of the referent. More info:
      https://kubernetes.io/docs/concepts/overview/working-with-
      objects/namespaces/ This is optional field, it gets defaulted to the
      object holding it if left out.
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    name = _messages.StringField(3)
    namespace = _messages.StringField(4)