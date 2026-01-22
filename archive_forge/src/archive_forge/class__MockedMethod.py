import difflib
import sys
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
class _MockedMethod(object):
    """A mocked API service method."""

    def __init__(self, key, mocked_client, real_method):
        self.__name__ = real_method.__name__
        self.__key = key
        self.__mocked_client = mocked_client
        self.__real_method = real_method
        self.method_config = real_method.method_config
        config = self.method_config()
        self.__request_type = getattr(self.__mocked_client.MESSAGES_MODULE, config.request_type_name)
        self.__response_type = getattr(self.__mocked_client.MESSAGES_MODULE, config.response_type_name)

    def _TypeCheck(self, msg, is_request):
        """Ensure the given message is of the expected type of this method.

        Args:
          msg: The message instance to check.
          is_request: True to validate against the expected request type,
             False to validate against the expected response type.

        Raises:
          exceptions.ConfigurationValueError: If the type of the message was
             not correct.
        """
        if is_request:
            mode = 'request'
            real_type = self.__request_type
        else:
            mode = 'response'
            real_type = self.__response_type
        if not isinstance(msg, real_type):
            raise exceptions.ConfigurationValueError('Expected {} is not of the correct type for method [{}].\n   Required: [{}]\n   Given:    [{}]'.format(mode, self.__key, real_type, type(msg)))

    def Expect(self, request, response=None, exception=None, enable_type_checking=True, **unused_kwargs):
        """Add an expectation on the mocked method.

        Exactly one of response and exception should be specified.

        Args:
          request: The request that should be expected
          response: The response that should be returned or None if
              exception is provided.
          exception: An exception that should be thrown, or None.
          enable_type_checking: When true, the message type of the request
              and response (if provided) will be checked against the types
              required by this method.
        """
        if enable_type_checking:
            self._TypeCheck(request, is_request=True)
            if response:
                self._TypeCheck(response, is_request=False)
        self.__mocked_client._request_responses.append(_ExpectedRequestResponse(self.__key, request, response=response, exception=exception))

    def __call__(self, request, **unused_kwargs):
        if self.__mocked_client._request_responses:
            request_response = self.__mocked_client._request_responses.pop(0)
        else:
            raise UnexpectedRequestException((self.__key, request), (None, None))
        response = request_response.ValidateAndRespond(self.__key, request)
        if response is None and self.__real_method:
            response = self.__real_method(request)
            print(encoding.MessageToRepr(response, multiline=True, shortstrings=True))
            return response
        return response