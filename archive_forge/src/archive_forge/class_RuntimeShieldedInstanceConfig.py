from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RuntimeShieldedInstanceConfig(_messages.Message):
    """A set of Shielded Instance options. See [Images using supported Shielded
  VM features](https://cloud.google.com/compute/docs/instances/modifying-
  shielded-vm). Not all combinations are valid.

  Fields:
    enableIntegrityMonitoring: Defines whether the instance has integrity
      monitoring enabled. Enables monitoring and attestation of the boot
      integrity of the instance. The attestation is performed against the
      integrity policy baseline. This baseline is initially derived from the
      implicitly trusted boot image when the instance is created. Enabled by
      default.
    enableSecureBoot: Defines whether the instance has Secure Boot enabled.
      Secure Boot helps ensure that the system only runs authentic software by
      verifying the digital signature of all boot components, and halting the
      boot process if signature verification fails. Disabled by default.
    enableVtpm: Defines whether the instance has the vTPM enabled. Enabled by
      default.
  """
    enableIntegrityMonitoring = _messages.BooleanField(1)
    enableSecureBoot = _messages.BooleanField(2)
    enableVtpm = _messages.BooleanField(3)