from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GraphqlResponse(_messages.Message):
    """The GraphQL response from Firebase Data Connect. It strives to match the
  GraphQL over HTTP spec. Note: Firebase Data Connect always responds with
  `Content-Type: application/json`. https://github.com/graphql/graphql-over-
  http/blob/main/spec/GraphQLOverHTTP.md#body

  Messages:
    DataValue: The result of the execution of the requested operation. If an
      error was raised before execution begins, the data entry should not be
      present in the result. (a request error:
      https://spec.graphql.org/draft/#sec-Errors.Request-Errors) If an error
      was raised during the execution that prevented a valid response, the
      data entry in the response should be null. (a field error:
      https://spec.graphql.org/draft/#sec-Errors.Error-Result-Format)

  Fields:
    data: The result of the execution of the requested operation. If an error
      was raised before execution begins, the data entry should not be present
      in the result. (a request error: https://spec.graphql.org/draft/#sec-
      Errors.Request-Errors) If an error was raised during the execution that
      prevented a valid response, the data entry in the response should be
      null. (a field error: https://spec.graphql.org/draft/#sec-Errors.Error-
      Result-Format)
    errors: Errors of this response. If the data entry in the response is not
      present, the errors entry must be present. It conforms to
      https://spec.graphql.org/draft/#sec-Errors.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DataValue(_messages.Message):
        """The result of the execution of the requested operation. If an error
    was raised before execution begins, the data entry should not be present
    in the result. (a request error: https://spec.graphql.org/draft/#sec-
    Errors.Request-Errors) If an error was raised during the execution that
    prevented a valid response, the data entry in the response should be null.
    (a field error: https://spec.graphql.org/draft/#sec-Errors.Error-Result-
    Format)

    Messages:
      AdditionalProperty: An additional property for a DataValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    data = _messages.MessageField('DataValue', 1)
    errors = _messages.MessageField('GraphqlError', 2, repeated=True)