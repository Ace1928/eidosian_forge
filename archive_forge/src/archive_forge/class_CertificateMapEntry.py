from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CertificateMapEntry(_messages.Message):
    """Defines a certificate map entry.

  Enums:
    MatcherValueValuesEnum: A predefined matcher for particular cases, other
      than SNI selection.
    StateValueValuesEnum: Output only. A serving state of this Certificate Map
      Entry.

  Messages:
    LabelsValue: Set of labels associated with a Certificate Map Entry.

  Fields:
    certificates: A set of Certificates defines for the given `hostname`.
      There can be defined up to fifteen certificates in each Certificate Map
      Entry. Each certificate must match pattern
      `projects/*/locations/*/certificates/*`.
    createTime: Output only. The creation timestamp of a Certificate Map
      Entry.
    description: One or more paragraphs of text description of a certificate
      map entry.
    hostname: A Hostname (FQDN, e.g. `example.com`) or a wildcard hostname
      expression (`*.example.com`) for a set of hostnames with common suffix.
      Used as Server Name Indication (SNI) for selecting a proper certificate.
    labels: Set of labels associated with a Certificate Map Entry.
    matcher: A predefined matcher for particular cases, other than SNI
      selection.
    name: A user-defined name of the Certificate Map Entry. Certificate Map
      Entry names must be unique globally and match pattern
      `projects/*/locations/*/certificateMaps/*/certificateMapEntries/*`.
    state: Output only. A serving state of this Certificate Map Entry.
    updateTime: Output only. The update timestamp of a Certificate Map Entry.
  """

    class MatcherValueValuesEnum(_messages.Enum):
        """A predefined matcher for particular cases, other than SNI selection.

    Values:
      MATCHER_UNSPECIFIED: A matcher has't been recognized.
      PRIMARY: A primary certificate that is served when SNI wasn't specified
        in the request or SNI couldn't be found in the map.
    """
        MATCHER_UNSPECIFIED = 0
        PRIMARY = 1

    class StateValueValuesEnum(_messages.Enum):
        """Output only. A serving state of this Certificate Map Entry.

    Values:
      SERVING_STATE_UNSPECIFIED: The status is undefined.
      ACTIVE: The configuration is serving.
      PENDING: Update is in progress. Some frontends may serve this
        configuration.
    """
        SERVING_STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PENDING = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Set of labels associated with a Certificate Map Entry.

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
    certificates = _messages.StringField(1, repeated=True)
    createTime = _messages.StringField(2)
    description = _messages.StringField(3)
    hostname = _messages.StringField(4)
    labels = _messages.MessageField('LabelsValue', 5)
    matcher = _messages.EnumField('MatcherValueValuesEnum', 6)
    name = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)
    updateTime = _messages.StringField(9)