from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareF5BigIpConfig(_messages.Message):
    """Represents configuration parameters for an F5 BIG-IP load balancer.

  Fields:
    address: The load balancer's IP address.
    partition: The preexisting partition to be used by the load balancer. This
      partition is usually created for the admin cluster for example:
      'my-f5-admin-partition'.
    snatPool: The pool name. Only necessary, if using SNAT.
  """
    address = _messages.StringField(1)
    partition = _messages.StringField(2)
    snatPool = _messages.StringField(3)