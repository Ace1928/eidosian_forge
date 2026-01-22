from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class LookupEffectiveGuestPolicyRequest(_messages.Message):
    """A request message for getting the effective guest policy assigned to the
  instance.

  Fields:
    osArchitecture: Architecture of OS running on the instance. The OS Config
      agent only provides this field for targeting if OS Inventory is enabled
      for that instance.
    osShortName: Short name of the OS running on the instance. The OS Config
      agent only provides this field for targeting if OS Inventory is enabled
      for that instance.
    osVersion: Version of the OS running on the instance. The OS Config agent
      only provides this field for targeting if OS Inventory is enabled for
      that VM instance.
  """
    osArchitecture = _messages.StringField(1)
    osShortName = _messages.StringField(2)
    osVersion = _messages.StringField(3)