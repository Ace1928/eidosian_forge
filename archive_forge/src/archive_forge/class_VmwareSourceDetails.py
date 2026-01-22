from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareSourceDetails(_messages.Message):
    """VmwareSourceDetails message describes a specific source details for the
  vmware source type.

  Fields:
    password: Input only. The credentials password. This is write only and can
      not be read in a GET operation.
    resolvedVcenterHost: The hostname of the vcenter.
    thumbprint: The thumbprint representing the certificate for the vcenter.
    username: The credentials username.
    vcenterIp: The ip address of the vcenter this Source represents.
  """
    password = _messages.StringField(1)
    resolvedVcenterHost = _messages.StringField(2)
    thumbprint = _messages.StringField(3)
    username = _messages.StringField(4)
    vcenterIp = _messages.StringField(5)