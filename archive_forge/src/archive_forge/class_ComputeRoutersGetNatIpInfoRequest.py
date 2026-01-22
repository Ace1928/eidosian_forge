from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComputeRoutersGetNatIpInfoRequest(_messages.Message):
    """A ComputeRoutersGetNatIpInfoRequest object.

  Fields:
    natName: Name of the nat service to filter the NAT IP information. If it
      is omitted, all nats for this router will be returned. Name should
      conform to RFC1035.
    project: Project ID for this request.
    region: Name of the region for this request.
    router: Name of the Router resource to query for Nat IP information. The
      name should conform to RFC1035.
  """
    natName = _messages.StringField(1)
    project = _messages.StringField(2, required=True)
    region = _messages.StringField(3, required=True)
    router = _messages.StringField(4, required=True)