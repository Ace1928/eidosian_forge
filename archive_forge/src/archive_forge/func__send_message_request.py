from __future__ import unicode_literals
import sys
import os
import requests
import requests.auth
import warnings
from winrm.exceptions import InvalidCredentialsError, WinRMError, WinRMTransportError
from winrm.encryption import Encryption
def _send_message_request(self, prepared_request, message):
    try:
        response = self.session.send(prepared_request, timeout=self.read_timeout_sec)
        response.raise_for_status()
        return response
    except requests.HTTPError as ex:
        if ex.response.status_code == 401:
            raise InvalidCredentialsError('the specified credentials were rejected by the server')
        if ex.response.content:
            response_text = self._get_message_response_text(ex.response)
        else:
            response_text = ''
        raise WinRMTransportError('http', ex.response.status_code, response_text)