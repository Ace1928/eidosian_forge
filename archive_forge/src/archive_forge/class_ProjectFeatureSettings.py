from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectFeatureSettings(_messages.Message):
    """ProjectFeatureSettings represents the features settings for the VM
  Manager. The project features settings can be set for a project.

  Enums:
    PatchAndConfigFeatureSetValueValuesEnum: Currently set
      PatchAndConfigFeatureSet for name.

  Fields:
    name: Required. Immutable. Name of the config, e.g.
      projects/12345/locations/global/projectFeatureSettings
    patchAndConfigFeatureSet: Currently set PatchAndConfigFeatureSet for name.
  """

    class PatchAndConfigFeatureSetValueValuesEnum(_messages.Enum):
        """Currently set PatchAndConfigFeatureSet for name.

    Values:
      PATCH_AND_CONFIG_FEATURE_SET_UNSPECIFIED: Not specified placeholder
      OSCONFIG_B: Basic feature set. Enables only the basic set of features.
      OSCONFIG_C: Classic set of functionality.
    """
        PATCH_AND_CONFIG_FEATURE_SET_UNSPECIFIED = 0
        OSCONFIG_B = 1
        OSCONFIG_C = 2
    name = _messages.StringField(1)
    patchAndConfigFeatureSet = _messages.EnumField('PatchAndConfigFeatureSetValueValuesEnum', 2)