from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1OASDocumentation(_messages.Message):
    """OpenAPI Specification documentation for a catalog item.

  Enums:
    FormatValueValuesEnum: Output only. The format of the input specification
      file contents.

  Fields:
    format: Output only. The format of the input specification file contents.
    spec: Required. The documentation file contents for the OpenAPI
      Specification. JSON and YAML file formats are supported.
  """

    class FormatValueValuesEnum(_messages.Enum):
        """Output only. The format of the input specification file contents.

    Values:
      FORMAT_UNSPECIFIED: The format is not available.
      YAML: YAML format.
      JSON: JSON format.
    """
        FORMAT_UNSPECIFIED = 0
        YAML = 1
        JSON = 2
    format = _messages.EnumField('FormatValueValuesEnum', 1)
    spec = _messages.MessageField('GoogleCloudApigeeV1DocumentationFile', 2)