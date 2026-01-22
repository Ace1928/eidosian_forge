from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParserConfig(_messages.Message):
    """The configuration for the parser. It determines how the server parses
  the messages.

  Enums:
    VersionValueValuesEnum: Immutable. Determines the version of both the
      default parser to be used when `schema` is not given, as well as the
      schematized parser used when `schema` is specified. This field is
      immutable after HL7v2 store creation.

  Fields:
    allowNullHeader: Determines whether messages with no header are allowed.
    segmentTerminator: Byte(s) to use as the segment terminator. If this is
      unset, '\\r' is used as the segment terminator, matching the HL7 version
      2 specification.
    version: Immutable. Determines the version of both the default parser to
      be used when `schema` is not given, as well as the schematized parser
      used when `schema` is specified. This field is immutable after HL7v2
      store creation.
  """

    class VersionValueValuesEnum(_messages.Enum):
        """Immutable. Determines the version of both the default parser to be
    used when `schema` is not given, as well as the schematized parser used
    when `schema` is specified. This field is immutable after HL7v2 store
    creation.

    Values:
      PARSER_VERSION_UNSPECIFIED: Unspecified parser version, equivalent to
        V1.
      V1: The `parsed_data` includes every given non-empty message field
        except the Field Separator (MSH-1) field. As a result, the parsed MSH
        segment starts with the MSH-2 field and the field numbers are off-by-
        one with respect to the HL7 standard.
      V2: The `parsed_data` includes every given non-empty message field.
      V3: This version is the same as V2, with the following change. The
        `parsed_data` contains unescaped escaped field separators, component
        separators, sub-component separators, repetition separators, escape
        characters, and truncation characters. If `schema` is specified, the
        schematized parser uses improved parsing heuristics compared to
        previous versions.
    """
        PARSER_VERSION_UNSPECIFIED = 0
        V1 = 1
        V2 = 2
        V3 = 3
    allowNullHeader = _messages.BooleanField(1)
    segmentTerminator = _messages.BytesField(2)
    version = _messages.EnumField('VersionValueValuesEnum', 3)