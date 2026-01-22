from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class APIVersionKind(_messages.Message):
    """APIVersionKind is an APIVersion and Kind tuple.

  Fields:
    apiVersion: APIVersion - the API version of the resource to watch.
    kind: Kind of the resource to watch. More info:
      https://git.k8s.io/community/contributors/devel/sig-architecture/api-
      conventions.md#types-kinds
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)