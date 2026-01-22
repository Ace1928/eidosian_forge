from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyOSFilter(_messages.Message):
    """Filtering criteria to select VMs based on OS details.

  Fields:
    osShortName: This should match OS short name emitted by the OS inventory
      agent. An empty value matches any OS.
    osVersion: This value should match the version emitted by the OS inventory
      agent. Prefix matches are supported if asterisk(*) is provided as the
      last character. For example, to match all versions with a major version
      of `7`, specify the following value for this field `7.*`
  """
    osShortName = _messages.StringField(1)
    osVersion = _messages.StringField(2)