from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ImageImportOsAdaptationParameters(_messages.Message):
    """Parameters affecting the OS adaptation process.

  Enums:
    LicenseTypeValueValuesEnum: Optional. Choose which type of license to
      apply to the imported image.

  Fields:
    generalize: Optional. Set to true in order to generalize the imported
      image. The generalization process enables co-existence of multiple VMs
      created from the same image. For Windows, generalizing the image removes
      computer-specific information such as installed drivers and the computer
      security identifier (SID).
    licenseType: Optional. Choose which type of license to apply to the
      imported image.
  """

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
    generalize = _messages.BooleanField(1)
    licenseType = _messages.EnumField('LicenseTypeValueValuesEnum', 2)