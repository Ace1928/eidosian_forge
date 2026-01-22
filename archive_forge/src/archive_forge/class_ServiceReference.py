from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ServiceReference(_messages.Message):
    """ServiceReference holds a reference to Service.legacy.k8s.io

  Fields:
    name: name is the name of the service. Required
    namespace: namespace is the namespace of the service.
    path: path is an optional URL path at which the webhook will be contacted.
    port: port is an optional service port at which the webhook will be
      contacted. `port` should be a valid port number (1-65535, inclusive).
      Defaults to 443 for backward compatibility.
  """
    name = _messages.StringField(1)
    namespace = _messages.StringField(2)
    path = _messages.StringField(3)
    port = _messages.IntegerField(4, variant=_messages.Variant.INT32)