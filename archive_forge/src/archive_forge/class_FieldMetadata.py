from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FieldMetadata(_messages.Message):
    """Specifies FHIR paths to match and how to handle the de-identification of
  matching fields.

  Enums:
    ActionValueValuesEnum: Deidentify action for one field.

  Fields:
    action: Deidentify action for one field.
    paths: List of paths to FHIR fields to redact. Each path is a period-
      separated list where each component is either a field name or FHIR type
      name. All types begin with an upper case letter. For example, the
      resource field "Patient.Address.city", which uses a string type, can be
      matched by "Patient.Address.String". Path also supports partial
      matching. For example, "Patient.Address.city" can be matched by
      "Address.city" (Patient omitted). Partial matching and type matching can
      be combined, for example "Patient.Address.city" can be matched by
      "Address.String". For "choice" types (those defined in the FHIR spec
      with the form: field[x]), use two separate components. For example,
      "deceasedAge.unit" is matched by "Deceased.Age.unit". Supported types
      are: AdministrativeGenderCode, Base64Binary, Boolean, Code, Date,
      DateTime, Decimal, HumanName, Id, Instant, Integer, LanguageCode,
      Markdown, Oid, PositiveInt, String, UnsignedInt, Uri, Uuid, Xhtml. The
      sub-type for HumanName(for example HumanName.given, HumanName.family)
      can be omitted.
  """

    class ActionValueValuesEnum(_messages.Enum):
        """Deidentify action for one field.

    Values:
      ACTION_UNSPECIFIED: No action specified.
      TRANSFORM: Transform the entire field based on transformations specified
        in TextConfig. When the specified transformation cannot be applied to
        a field (for example, a Crypto Hash transformation cannot be applied
        to a FHIR Date field), RedactConfig is used.
      INSPECT_AND_TRANSFORM: Inspect and transform any found PHI. When
        `AnnotationConfig` is provided, annotations of PHI are generated,
        except for Date and Datetime.
      DO_NOT_TRANSFORM: Do not transform.
    """
        ACTION_UNSPECIFIED = 0
        TRANSFORM = 1
        INSPECT_AND_TRANSFORM = 2
        DO_NOT_TRANSFORM = 3
    action = _messages.EnumField('ActionValueValuesEnum', 1)
    paths = _messages.StringField(2, repeated=True)