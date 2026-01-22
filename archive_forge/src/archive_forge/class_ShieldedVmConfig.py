from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShieldedVmConfig(_messages.Message):
    """A set of Shielded VM options.

  Fields:
    enableIntegrityMonitoring: Defines whether the instance has integrity
      monitoring enabled.
    enableSecureBoot: Defines whether the instance has Secure Boot enabled.
    enableVtpm: Defines whether the instance has the vTPM enabled.
  """
    enableIntegrityMonitoring = _messages.BooleanField(1)
    enableSecureBoot = _messages.BooleanField(2)
    enableVtpm = _messages.BooleanField(3)