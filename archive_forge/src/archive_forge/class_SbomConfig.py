from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SbomConfig(_messages.Message):
    """Config for whether to generate SBOMs for resources in this repository,
  as well as output fields describing current state.

  Enums:
    EnablementConfigValueValuesEnum: Optional. Config for whether this
      repository has sbom generation disabled.
    EnablementStateValueValuesEnum: Output only. State of feature enablement,
      combining repository enablement config and API enablement state.

  Fields:
    enablementConfig: Optional. Config for whether this repository has sbom
      generation disabled.
    enablementState: Output only. State of feature enablement, combining
      repository enablement config and API enablement state.
    enablementStateReason: Output only. Reason for the repository state and
      potential actions to activate it.
    gcsBucket: Optional. The GCS bucket to put the generated SBOMs into.
    lastEnableTime: Output only. The last time this repository config was set
      to INHERITED.
  """

    class EnablementConfigValueValuesEnum(_messages.Enum):
        """Optional. Config for whether this repository has sbom generation
    disabled.

    Values:
      ENABLEMENT_CONFIG_UNSPECIFIED: Unspecified config was not set. This will
        be interpreted as DISABLED.
      INHERITED: Inherited indicates the repository is allowed for SBOM
        generation, however the actual state will be inherited from the API
        enablement state.
      DISABLED: Disabled indicates the repository will not generate SBOMs.
    """
        ENABLEMENT_CONFIG_UNSPECIFIED = 0
        INHERITED = 1
        DISABLED = 2

    class EnablementStateValueValuesEnum(_messages.Enum):
        """Output only. State of feature enablement, combining repository
    enablement config and API enablement state.

    Values:
      ENABLEMENT_STATE_UNSPECIFIED: Enablement state is unclear.
      SBOM_UNSUPPORTED: Repository does not support SBOM generation.
      SBOM_DISABLED: SBOM generation is disabled for this repository.
      SBOM_ACTIVE: SBOM generation is active for this feature.
    """
        ENABLEMENT_STATE_UNSPECIFIED = 0
        SBOM_UNSUPPORTED = 1
        SBOM_DISABLED = 2
        SBOM_ACTIVE = 3
    enablementConfig = _messages.EnumField('EnablementConfigValueValuesEnum', 1)
    enablementState = _messages.EnumField('EnablementStateValueValuesEnum', 2)
    enablementStateReason = _messages.StringField(3)
    gcsBucket = _messages.StringField(4)
    lastEnableTime = _messages.StringField(5)