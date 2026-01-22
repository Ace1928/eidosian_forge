from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2FulfillmentGenericWebService(_messages.Message):
    """Represents configuration for a generic web service. Dialogflow supports
  two mechanisms for authentications: - Basic authentication with username and
  password. - Authentication with additional authentication headers. More
  information could be found at:
  https://cloud.google.com/dialogflow/docs/fulfillment-configure.

  Messages:
    RequestHeadersValue: Optional. The HTTP request headers to send together
      with fulfillment requests.

  Fields:
    isCloudFunction: Optional. Indicates if generic web service is created
      through Cloud Functions integration. Defaults to false.
      is_cloud_function is deprecated. Cloud functions can be configured by
      its uri as a regular web service now.
    password: Optional. The password for HTTP Basic authentication.
    requestHeaders: Optional. The HTTP request headers to send together with
      fulfillment requests.
    uri: Required. The fulfillment URI for receiving POST requests. It must
      use https protocol.
    username: Optional. The user name for HTTP Basic authentication.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class RequestHeadersValue(_messages.Message):
        """Optional. The HTTP request headers to send together with fulfillment
    requests.

    Messages:
      AdditionalProperty: An additional property for a RequestHeadersValue
        object.

    Fields:
      additionalProperties: Additional properties of type RequestHeadersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a RequestHeadersValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    isCloudFunction = _messages.BooleanField(1)
    password = _messages.StringField(2)
    requestHeaders = _messages.MessageField('RequestHeadersValue', 3)
    uri = _messages.StringField(4)
    username = _messages.StringField(5)