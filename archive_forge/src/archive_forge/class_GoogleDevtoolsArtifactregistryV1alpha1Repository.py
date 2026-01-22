from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleDevtoolsArtifactregistryV1alpha1Repository(_messages.Message):
    """A Repository for storing artifacts with a specific format.

  Enums:
    FormatValueValuesEnum: Optional. The format of packages that are stored in
      the repository.
    ModeValueValuesEnum: Optional. The mode of the repository.

  Messages:
    LabelsValue: Labels with user-defined metadata. This field may contain up
      to 64 entries. Label keys and values may be no longer than 63
      characters. Label keys must begin with a lowercase letter and may only
      contain lowercase letters, numeric characters, underscores, and dashes.

  Fields:
    createTime: Output only. The time when the repository was created.
    description: The user-provided description of the repository.
    format: Optional. The format of packages that are stored in the
      repository.
    kmsKeyName: The Cloud KMS resource name of the customer managed encryption
      key that's used to encrypt the contents of the Repository. Has the form:
      `projects/my-project/locations/my-region/keyRings/my-kr/cryptoKeys/my-
      key`. This value may not be changed after the Repository has been
      created.
    labels: Labels with user-defined metadata. This field may contain up to 64
      entries. Label keys and values may be no longer than 63 characters.
      Label keys must begin with a lowercase letter and may only contain
      lowercase letters, numeric characters, underscores, and dashes.
    mode: Optional. The mode of the repository.
    name: The name of the repository, for example: `projects/p1/locations/us-
      central1/repositories/repo1`.
    satisfiesPzs: Output only. If set, the repository satisfies physical zone
      separation.
    sizeBytes: Output only. The size, in bytes, of all artifact storage in
      this repository. Repositories that are generally available or in public
      preview use this to calculate storage costs.
    updateTime: Output only. The time when the repository was last updated.
  """

    class FormatValueValuesEnum(_messages.Enum):
        """Optional. The format of packages that are stored in the repository.

    Values:
      FORMAT_UNSPECIFIED: Unspecified package format.
      DOCKER: Docker package format.
      MAVEN: Maven package format.
      NPM: NPM package format.
      APT: APT package format.
      YUM: YUM package format.
      GOOGET: GooGet package format.
      PYTHON: Python package format.
    """
        FORMAT_UNSPECIFIED = 0
        DOCKER = 1
        MAVEN = 2
        NPM = 3
        APT = 4
        YUM = 5
        GOOGET = 6
        PYTHON = 7

    class ModeValueValuesEnum(_messages.Enum):
        """Optional. The mode of the repository.

    Values:
      MODE_UNSPECIFIED: Unspecified mode.
      STANDARD_REPOSITORY: A standard repository storing artifacts.
      VIRTUAL_REPOSITORY: A virtual repository to serve artifacts from one or
        more sources.
    """
        MODE_UNSPECIFIED = 0
        STANDARD_REPOSITORY = 1
        VIRTUAL_REPOSITORY = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels with user-defined metadata. This field may contain up to 64
    entries. Label keys and values may be no longer than 63 characters. Label
    keys must begin with a lowercase letter and may only contain lowercase
    letters, numeric characters, underscores, and dashes.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    createTime = _messages.StringField(1)
    description = _messages.StringField(2)
    format = _messages.EnumField('FormatValueValuesEnum', 3)
    kmsKeyName = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    mode = _messages.EnumField('ModeValueValuesEnum', 6)
    name = _messages.StringField(7)
    satisfiesPzs = _messages.BooleanField(8)
    sizeBytes = _messages.IntegerField(9)
    updateTime = _messages.StringField(10)