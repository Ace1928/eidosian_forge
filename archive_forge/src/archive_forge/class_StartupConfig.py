from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StartupConfig(_messages.Message):
    """Configuration to handle the startup of instances during cluster create
  and update process.

  Fields:
    requiredRegistrationFraction: Optional. The config setting to enable
      cluster creation/ updation to be successful only after
      required_registration_fraction of instances are up and running. This
      configuration is applicable to only secondary workers for now. The
      cluster will fail if required_registration_fraction of instances are not
      available. This will include instance creation, agent registration, and
      service registration (if enabled).
  """
    requiredRegistrationFraction = _messages.FloatField(1)