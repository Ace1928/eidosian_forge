from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminOsEnvironmentConfig(_messages.Message):
    """Specifies operating system operation settings for cluster provisioning.

  Fields:
    packageRepoExcluded: Whether the package repo should be added when
      initializing bare metal machines.
  """
    packageRepoExcluded = _messages.BooleanField(1)