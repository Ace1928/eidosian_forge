from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LicenseTypeValueValuesEnum(_messages.Enum):
    """Optional. Choose which type of license to apply to the imported image.

    Values:
      COMPUTE_ENGINE_LICENSE_TYPE_DEFAULT: The license type is the default for
        the OS.
      COMPUTE_ENGINE_LICENSE_TYPE_PAYG: The license type is Pay As You Go
        license type.
      COMPUTE_ENGINE_LICENSE_TYPE_BYOL: The license type is Bring Your Own
        License type.
    """
    COMPUTE_ENGINE_LICENSE_TYPE_DEFAULT = 0
    COMPUTE_ENGINE_LICENSE_TYPE_PAYG = 1
    COMPUTE_ENGINE_LICENSE_TYPE_BYOL = 2