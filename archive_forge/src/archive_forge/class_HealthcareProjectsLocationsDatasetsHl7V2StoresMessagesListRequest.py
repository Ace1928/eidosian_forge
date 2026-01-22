from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesListRequest(_messages.Message):
    """A HealthcareProjectsLocationsDatasetsHl7V2StoresMessagesListRequest
  object.

  Enums:
    ViewValueValuesEnum: Specifies the parts of the Message to return in the
      response. When unspecified, equivalent to BASIC. Setting this to
      anything other than BASIC with a `page_size` larger than the default can
      generate a large response, which impacts the performance of this method.

  Fields:
    filter: Restricts messages returned to those matching a filter. The
      following syntax is available: * A string field value can be written as
      text inside quotation marks, for example `"query text"`. The only valid
      relational operation for text fields is equality (`=`), where text is
      searched within the field, rather than having the field be equal to the
      text. For example, `"Comment = great"` returns messages with `great` in
      the comment field. * A number field value can be written as an integer,
      a decimal, or an exponential. The valid relational operators for number
      fields are the equality operator (`=`), along with the less than/greater
      than operators (`<`, `<=`, `>`, `>=`). Note that there is no inequality
      (`!=`) operator. You can prepend the `NOT` operator to an expression to
      negate it. * A date field value must be written in `yyyy-mm-dd` form.
      Fields with date and time use the RFC3339 time format. Leading zeros are
      required for one-digit months and days. The valid relational operators
      for date fields are the equality operator (`=`) , along with the less
      than/greater than operators (`<`, `<=`, `>`, `>=`). Note that there is
      no inequality (`!=`) operator. You can prepend the `NOT` operator to an
      expression to negate it. * Multiple field query expressions can be
      combined in one query by adding `AND` or `OR` operators between the
      expressions. If a boolean operator appears within a quoted string, it is
      not treated as special, it's just another part of the character string
      to be matched. You can prepend the `NOT` operator to an expression to
      negate it. Fields/functions available for filtering are: *
      `message_type`, from the MSH-9.1 field. For example, `NOT message_type =
      "ADT"`. * `send_date` or `sendDate`, the YYYY-MM-DD date the message was
      sent in the dataset's time_zone, from the MSH-7 segment. For example,
      `send_date < "2017-01-02"`. * `send_time`, the timestamp when the
      message was sent, using the RFC3339 time format for comparisons, from
      the MSH-7 segment. For example, `send_time <
      "2017-01-02T00:00:00-05:00"`. * `create_time`, the timestamp when the
      message was created in the HL7v2 store. Use the RFC3339 time format for
      comparisons. For example, `create_time < "2017-01-02T00:00:00-05:00"`. *
      `send_facility`, the care center that the message came from, from the
      MSH-4 segment. For example, `send_facility = "ABC"`. * `PatientId(value,
      type)`, which matches if the message lists a patient having an ID of the
      given value and type in the PID-2, PID-3, or PID-4 segments. For
      example, `PatientId("123456", "MRN")`. * `labels.x`, a string value of
      the label with key `x` as set using the Message.labels map. For example,
      `labels."priority"="high"`. The operator `:*` can be used to assert the
      existence of a label. For example, `labels."priority":*`.
    orderBy: Orders messages returned by the specified order_by clause.
      Syntax:
      https://cloud.google.com/apis/design/design_patterns#sorting_order
      Fields available for ordering are: * `send_time`
    pageSize: Limit on the number of messages to return in a single response.
      If not specified, 100 is used. May not be larger than 1000.
    pageToken: The next_page_token value returned from the previous List
      request, if any.
    parent: Name of the HL7v2 store to retrieve messages from.
    view: Specifies the parts of the Message to return in the response. When
      unspecified, equivalent to BASIC. Setting this to anything other than
      BASIC with a `page_size` larger than the default can generate a large
      response, which impacts the performance of this method.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies the parts of the Message to return in the response. When
    unspecified, equivalent to BASIC. Setting this to anything other than
    BASIC with a `page_size` larger than the default can generate a large
    response, which impacts the performance of this method.

    Values:
      MESSAGE_VIEW_UNSPECIFIED: Not specified, equivalent to FULL.
      RAW_ONLY: Server responses include all the message fields except
        parsed_data field, and schematized_data fields.
      PARSED_ONLY: Server responses include all the message fields except data
        field, and schematized_data fields.
      FULL: Server responses include all the message fields.
      SCHEMATIZED_ONLY: Server responses include all the message fields except
        data and parsed_data fields.
      BASIC: Server responses include only the name field.
    """
        MESSAGE_VIEW_UNSPECIFIED = 0
        RAW_ONLY = 1
        PARSED_ONLY = 2
        FULL = 3
        SCHEMATIZED_ONLY = 4
        BASIC = 5
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 6)