import abc
import base64
import enum
import json
import six
from google.auth import exceptions
def _inject_authenticated_request_body(self, request_body):
    if self._client_authentication is not None and self._client_authentication.client_auth_type is ClientAuthType.request_body:
        if request_body is None:
            raise exceptions.OAuthError('HTTP request does not support request-body')
        else:
            request_body['client_id'] = self._client_authentication.client_id
            request_body['client_secret'] = self._client_authentication.client_secret or ''