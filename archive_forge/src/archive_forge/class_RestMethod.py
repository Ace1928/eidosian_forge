from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RestMethod(_messages.Message):
    """A RestMethod object.

  Messages:
    MediaUploadValue: Media upload parameters.
    ParametersValue: Details for all parameters in this method.
    RequestValue: The schema for the request.
    ResponseValue: The schema for the response.

  Fields:
    description: Description of this method.
    etagRequired: Whether this method requires an ETag to be specified. The
      ETag is sent as an HTTP If-Match or If-None-Match header.
    httpMethod: HTTP method used by this method.
    id: A unique ID for this method. This property can be used to match
      methods between different versions of Discovery.
    mediaUpload: Media upload parameters.
    parameterOrder: Ordered list of required parameters, serves as a hint to
      clients on how to structure their method signatures. The array is
      ordered such that the "most-significant" parameter appears first.
    parameters: Details for all parameters in this method.
    path: The URI path of this REST method. Should be used in conjunction with
      the basePath property at the api-level.
    request: The schema for the request.
    response: The schema for the response.
    scopes: OAuth 2.0 scopes applicable to this method.
    supportsMediaDownload: Whether this method supports media downloads.
    supportsMediaUpload: Whether this method supports media uploads.
    supportsSubscription: Whether this method supports subscriptions.
  """

    class MediaUploadValue(_messages.Message):
        """Media upload parameters.

    Messages:
      ProtocolsValue: Supported upload protocols.

    Fields:
      accept: MIME Media Ranges for acceptable media uploads to this method.
      maxSize: Maximum size of a media upload, such as "1MB", "2GB" or "3TB".
      protocols: Supported upload protocols.
    """

        class ProtocolsValue(_messages.Message):
            """Supported upload protocols.

      Messages:
        ResumableValue: Supports the Resumable Media Upload protocol.
        SimpleValue: Supports uploading as a single HTTP request.

      Fields:
        resumable: Supports the Resumable Media Upload protocol.
        simple: Supports uploading as a single HTTP request.
      """

            class ResumableValue(_messages.Message):
                """Supports the Resumable Media Upload protocol.

        Fields:
          multipart: True if this endpoint supports uploading multipart media.
          path: The URI path to be used for upload. Should be used in
            conjunction with the basePath property at the api-level.
        """
                multipart = _messages.BooleanField(1, default=True)
                path = _messages.StringField(2)

            class SimpleValue(_messages.Message):
                """Supports uploading as a single HTTP request.

        Fields:
          multipart: True if this endpoint supports upload multipart media.
          path: The URI path to be used for upload. Should be used in
            conjunction with the basePath property at the api-level.
        """
                multipart = _messages.BooleanField(1, default=True)
                path = _messages.StringField(2)
            resumable = _messages.MessageField('ResumableValue', 1)
            simple = _messages.MessageField('SimpleValue', 2)
        accept = _messages.StringField(1, repeated=True)
        maxSize = _messages.StringField(2)
        protocols = _messages.MessageField('ProtocolsValue', 3)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """Details for all parameters in this method.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Details for a single parameter in this method.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A JsonSchema attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('JsonSchema', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    class RequestValue(_messages.Message):
        """The schema for the request.

    Fields:
      _ref: Schema ID for the request schema.
    """
        _ref = _messages.StringField(1)

    class ResponseValue(_messages.Message):
        """The schema for the response.

    Fields:
      _ref: Schema ID for the response schema.
    """
        _ref = _messages.StringField(1)
    description = _messages.StringField(1)
    etagRequired = _messages.BooleanField(2)
    httpMethod = _messages.StringField(3)
    id = _messages.StringField(4)
    mediaUpload = _messages.MessageField('MediaUploadValue', 5)
    parameterOrder = _messages.StringField(6, repeated=True)
    parameters = _messages.MessageField('ParametersValue', 7)
    path = _messages.StringField(8)
    request = _messages.MessageField('RequestValue', 9)
    response = _messages.MessageField('ResponseValue', 10)
    scopes = _messages.StringField(11, repeated=True)
    supportsMediaDownload = _messages.BooleanField(12)
    supportsMediaUpload = _messages.BooleanField(13)
    supportsSubscription = _messages.BooleanField(14)