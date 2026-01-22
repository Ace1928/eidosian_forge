from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsCloudbuildV1RepoSource(_messages.Message):
    """Location of the source in a Google Cloud Source Repository.

  Messages:
    SubstitutionsValue: Substitutions to use in a triggered build. Should only
      be used with RunBuildTrigger

  Fields:
    branchName: Regex matching branches to build. The syntax of the regular
      expressions accepted is the syntax accepted by RE2 and described at
      https://github.com/google/re2/wiki/Syntax
    commitSha: Explicit commit SHA to build.
    dir: Directory, relative to the source root, in which to run the build.
      This must be a relative path. If a step's `dir` is specified and is an
      absolute path, this value is ignored for that step's execution.
    invertRegex: Only trigger a build if the revision regex does NOT match the
      revision regex.
    projectId: ID of the project that owns the Cloud Source Repository. If
      omitted, the project ID requesting the build is assumed.
    repoName: Name of the Cloud Source Repository.
    substitutions: Substitutions to use in a triggered build. Should only be
      used with RunBuildTrigger
    tagName: Regex matching tags to build. The syntax of the regular
      expressions accepted is the syntax accepted by RE2 and described at
      https://github.com/google/re2/wiki/Syntax
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class SubstitutionsValue(_messages.Message):
        """Substitutions to use in a triggered build. Should only be used with
    RunBuildTrigger

    Messages:
      AdditionalProperty: An additional property for a SubstitutionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type SubstitutionsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a SubstitutionsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    branchName = _messages.StringField(1)
    commitSha = _messages.StringField(2)
    dir = _messages.StringField(3)
    invertRegex = _messages.BooleanField(4)
    projectId = _messages.StringField(5)
    repoName = _messages.StringField(6)
    substitutions = _messages.MessageField('SubstitutionsValue', 7)
    tagName = _messages.StringField(8)