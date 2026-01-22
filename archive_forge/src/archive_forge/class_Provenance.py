from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Provenance(_messages.Message):
    """Provenance configuration.

  Enums:
    EnabledValueValuesEnum: Optional. Provenance push mode.
    RegionValueValuesEnum: Optional. Provenance region.
    StorageValueValuesEnum: Optional. Where provenance is stored.

  Fields:
    enabled: Optional. Provenance push mode.
    region: Optional. Provenance region.
    storage: Optional. Where provenance is stored.
  """

    class EnabledValueValuesEnum(_messages.Enum):
        """Optional. Provenance push mode.

    Values:
      ENABLED_UNSPECIFIED: Default to disabled (before AA regionalization),
        optimistic after
      REQUIRED: Provenance failures would fail the run
      OPTIMISTIC: GCB will attempt to push to artifact analaysis and build
        state would not be impacted by the push failures.
      DISABLED: Disable the provenance push entirely.
    """
        ENABLED_UNSPECIFIED = 0
        REQUIRED = 1
        OPTIMISTIC = 2
        DISABLED = 3

    class RegionValueValuesEnum(_messages.Enum):
        """Optional. Provenance region.

    Values:
      REGION_UNSPECIFIED: The PipelineRun/TaskRun/Workflow will be rejected.
        Update this comment to push to the same region as the run in Artifact
        Analysis when it's regionalized.
      GLOBAL: Push provenance to Artifact Analysis in global region.
    """
        REGION_UNSPECIFIED = 0
        GLOBAL = 1

    class StorageValueValuesEnum(_messages.Enum):
        """Optional. Where provenance is stored.

    Values:
      STORAGE_UNSPECIFIED: Default PREFER_ARTIFACT_PROJECT.
      PREFER_ARTIFACT_PROJECT: GCB will attempt to push provenance to the
        artifact project. If it is not available, fallback to build project.
      ARTIFACT_PROJECT_ONLY: Only push to artifact project.
      BUILD_PROJECT_ONLY: Only push to build project.
    """
        STORAGE_UNSPECIFIED = 0
        PREFER_ARTIFACT_PROJECT = 1
        ARTIFACT_PROJECT_ONLY = 2
        BUILD_PROJECT_ONLY = 3
    enabled = _messages.EnumField('EnabledValueValuesEnum', 1)
    region = _messages.EnumField('RegionValueValuesEnum', 2)
    storage = _messages.EnumField('StorageValueValuesEnum', 3)