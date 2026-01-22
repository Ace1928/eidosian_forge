from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckUpgradeResponse(_messages.Message):
    """Message containing information about the result of an upgrade check
  operation.

  Enums:
    ContainsPypiModulesConflictValueValuesEnum: Output only. Whether build has
      succeeded or failed on modules conflicts.

  Messages:
    PypiDependenciesValue: Pypi dependencies specified in the environment
      configuration, at the time when the build was triggered.

  Fields:
    buildLogUri: Output only. Url for a docker build log of an upgraded image.
    containsPypiModulesConflict: Output only. Whether build has succeeded or
      failed on modules conflicts.
    imageVersion: Composer image for which the build was happening.
    pypiConflictBuildLogExtract: Output only. Extract from a docker image
      build log containing information about pypi modules conflicts.
    pypiDependencies: Pypi dependencies specified in the environment
      configuration, at the time when the build was triggered.
  """

    class ContainsPypiModulesConflictValueValuesEnum(_messages.Enum):
        """Output only. Whether build has succeeded or failed on modules
    conflicts.

    Values:
      CONFLICT_RESULT_UNSPECIFIED: It is unknown whether build had conflicts
        or not.
      CONFLICT: There were python packages conflicts.
      NO_CONFLICT: There were no python packages conflicts.
    """
        CONFLICT_RESULT_UNSPECIFIED = 0
        CONFLICT = 1
        NO_CONFLICT = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PypiDependenciesValue(_messages.Message):
        """Pypi dependencies specified in the environment configuration, at the
    time when the build was triggered.

    Messages:
      AdditionalProperty: An additional property for a PypiDependenciesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        PypiDependenciesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PypiDependenciesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    buildLogUri = _messages.StringField(1)
    containsPypiModulesConflict = _messages.EnumField('ContainsPypiModulesConflictValueValuesEnum', 2)
    imageVersion = _messages.StringField(3)
    pypiConflictBuildLogExtract = _messages.StringField(4)
    pypiDependencies = _messages.MessageField('PypiDependenciesValue', 5)